#!/usr/bin/env bash
#
# DSV4 CP / throughput 对照测试 (Ascend NPU, tp=16)
# ---------------------------------------------------------------------------
# 目的: 用"单变量对照"回答两个问题
#   1) CP 到底在什么场景有收益 (低并发长序列 prefill), scaling 如何?
#   2) 高并发吞吐场景 DP 是否确实优于 CP?
#
# 设计原则 (避免上次那种"多变量一起变"导致结论不可比):
#   - 固定 tp=16、chunked-prefill-size、spec 算法、mem-fraction 不变;
#   - 每个并行配置满足 dp * cp == tp (注意力全并行), 只挪 dp<->cp;
#   - 用 SPEC=off 先拿"干净"的算子级数字, 再用 SPEC=eagle 看真实收益;
#   - 分离 prefill / decode 两个 regime, 别用 in=128k/out=1k 这种混合workload下结论.
#
# 用法:
#   export MODEL_PATH=/path/to/dsv4
#   export DATASET=/data/cjr/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
#   # 跑全部实验 (spec 关闭, 干净数字):
#   SPEC=off bash benchmark/dsv4_cp_bench.sh all
#   # 只跑某个实验:
#   bash benchmark/dsv4_cp_bench.sh S1     # CP prefill scaling
#   bash benchmark/dsv4_cp_bench.sh S3     # 吞吐 regime: DP vs CP
#
# 结果: $RESULT_DIR/summary.csv  (tag, total_tok_s, out_tok_s, ttft_ms, tpot_ms, accept)
# ---------------------------------------------------------------------------
set -uo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
DATASET="${DATASET:-/data/cjr/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"
HOST=127.0.0.1
PORT="${PORT:-9527}"
TP=16
STRAT="${STRAT:-interleave}"          # interleave (round-robin) 或 zigzag (in-seq)
RESULT_DIR="${RESULT_DIR:-./cp_bench_results}"
mkdir -p "$RESULT_DIR"
SUMMARY="$RESULT_DIR/summary.csv"
[[ -f "$SUMMARY" ]] || echo "tag,total_tok_s,out_tok_s,ttft_ms,tpot_ms,accept" > "$SUMMARY"

# ---- 固定的公共 launch 参数 (所有 run 一致, 保证可比) ----------------------
COMMON_LAUNCH=(
  --model-path "$MODEL_PATH"
  --page-size 128
  --tp-size "$TP"
  --trust-remote-code
  --device npu
  --attention-backend ascend
  --watchdog-timeout 9000
  --host "$HOST" --port "$PORT"
  --mem-fraction-static 0.95
  --disable-radix-cache
  --max-running-requests 32
  --moe-a2a-backend deepep --deepep-mode auto
  --quantization modelslim --enable-dp-lm-head
  --kv-cache-dtype auto
  --chunked-prefill-size 65536      # 固定! 两侧都一样, 别再一个 131072 一个 65536
)

spec_flags() {   # SPEC=off 关闭投机解码 (拿干净的 prefill/decode 数字)
  if [[ "${SPEC:-eagle}" == "off" ]]; then
    echo ""
  else
    echo "--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3"
  fi
}

wait_health() {
  for _ in $(seq 1 720); do
    curl -sf "http://$HOST:$PORT/health_generate" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "!! server 未就绪"; return 1
}

SERVER_PID=""
launch() {   # dp cp
  local dp=$1 cp=$2
  local cp_flags=()
  [[ "$cp" -gt 1 ]] && cp_flags=(--enable-prefill-cp --cp-strategy "$STRAT")
  echo ">>> LAUNCH dp=$dp cp=$cp strat=$STRAT spec=${SPEC:-eagle}"
  python3 -m sglang.launch_server "${COMMON_LAUNCH[@]}" \
    --dp-size "$dp" --enable-dp-attention \
    $(spec_flags) "${cp_flags[@]}" \
    > "$RESULT_DIR/server_dp${dp}_cp${cp}.log" 2>&1 &
  SERVER_PID=$!
  wait_health || { kill "$SERVER_PID" 2>/dev/null; return 1; }
}

kill_server() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  sleep 15                          # 等 NPU 显存/HCCL 释放
  SERVER_PID=""
}

bench() {   # tag conc nprompts inlen outlen
  local tag=$1 conc=$2 np=$3 inlen=$4 outlen=$5
  local out="$RESULT_DIR/bench_${tag}.log"
  echo ">>> BENCH $tag  conc=$conc np=$np in=$inlen out=$outlen"
  python3 -m sglang.bench_serving \
    --backend sglang --host "$HOST" --port "$PORT" \
    --dataset-name random --dataset-path "$DATASET" \
    --num-prompts "$np" --max-concurrency "$conc" \
    --random-input-len "$inlen" --random-output-len "$outlen" \
    --random-range-ratio 1 --warmup-requests 0 \
    2>&1 | tee "$out"
  awk -v tag="$tag" '
    /Total token throughput/     {tt=$NF}
    /Output token throughput/    {ot=$NF}
    /Mean TTFT/                  {ttft=$NF}
    /Mean TPOT/                  {tpot=$NF}
    /Accept length/             {acc=$NF}
    END {printf "%s,%s,%s,%s,%s,%s\n", tag, tt, ot, ttft, tpot, acc}' "$out" >> "$SUMMARY"
}

# 并行配置 (dp cp), 均满足 dp*cp=16 -----------------------------------------
#   P0 = 纯 DP baseline; P1 = 用户已跑的 CP; P2/P3 = 加大 cp 看 scaling
P0=(16 1); P1=(8 2); P2=(4 4); P3=(1 16)

# ===========================================================================
# S1 — CP prefill scaling (低并发长序列, CP 的主场)
#   固定 concurrency=1, out=8 (几乎只测 prefill/TTFT), 扫 cp_size
#   期望: TTFT 随 cp 增大而下降. 若不降 => 瓶颈在 compressor all-gather.
# ===========================================================================
run_S1() {
  for cfg in P0 P1 P2 P3; do
    eval "dp=\${${cfg}[0]}; cp=\${${cfg}[1]}"
    launch "$dp" "$cp" || { kill_server; continue; }
    bench "S1_${cfg}_dp${dp}cp${cp}" 1 4 131072 8
    kill_server
  done
}

# ===========================================================================
# S2 — CP 的 decode 开销 (隔离 TPOT)
#   中等长度 in=4096, out=1024, concurrency=1, 只看 TPOT.
#   量化 CP 把 KV 分散到多卡后, 每个 decode step 的通信税.
# ===========================================================================
run_S2() {
  for cfg in P0 P1 P3; do
    eval "dp=\${${cfg}[0]}; cp=\${${cfg}[1]}"
    launch "$dp" "$cp" || { kill_server; continue; }
    bench "S2_${cfg}_dp${dp}cp${cp}" 1 8 4096 1024
    kill_server
  done
}

# ===========================================================================
# S3 — 吞吐 regime (为什么这里 DP 赢)
#   concurrency=16, 真实 workload, P0(纯DP) vs P1(CP). 看 total throughput.
# ===========================================================================
run_S3() {
  for cfg in P0 P1; do
    eval "dp=\${${cfg}[0]}; cp=\${${cfg}[1]}"
    launch "$dp" "$cp" || { kill_server; continue; }
    bench "S3_${cfg}_dp${dp}cp${cp}" 16 48 131072 1024
    kill_server
  done
}

# ===========================================================================
# S4 — 并发拐点 (纯 DP 的真正调优旋钮)
#   P0 固定, 扫 concurrency, 找 "吞吐饱和 / 延迟开始恶化" 的拐点.
#   这决定你真实部署应该把 max-concurrency 设多少.
# ===========================================================================
run_S4() {
  local dp=${P0[0]} cp=${P0[1]}
  launch "$dp" "$cp" || { kill_server; return; }
  for conc in 1 2 4 8 16 32; do
    bench "S4_conc${conc}" "$conc" $((conc*3)) 131072 1024
  done
  kill_server
}

case "${1:-all}" in
  S1) run_S1 ;;
  S2) run_S2 ;;
  S3) run_S3 ;;
  S4) run_S4 ;;
  all) run_S1; run_S2; run_S3; run_S4 ;;
  *) echo "usage: $0 {S1|S2|S3|S4|all}"; exit 1 ;;
esac

echo "==== 汇总 ===="; column -t -s, "$SUMMARY"
