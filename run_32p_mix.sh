#!/usr/bin/env bash

set -euo pipefail

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

SGLANG_ROOT=${SGLANG_ROOT:-/home/hanwlax/workspace/sglang}
MODEL_PATH=${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-8cards-quarot-all-0722}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/home/weights/DSpark-Kimi-K3-yi}
MASTER_ADDR=${MASTER_ADDR:-192.168.25.213}
MASTER_PORT=${MASTER_PORT:-5000}
SERVER_PORT=${SERVER_PORT:-30000}
COMM_IF=${COMM_IF:-enp196s0f0}

# Conservative defaults for the first 4-node DSpark/MTP smoke test.
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.72}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-32768}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
ENABLE_NPU_GRAPH=${ENABLE_NPU_GRAPH:-0}
ENABLE_RADIX_CACHE=${ENABLE_RADIX_CACHE:-0}

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=2000
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export SGLANG_RAGGED_VERIFY_MODE=static
export PYTHONUNBUFFERED=1
unset SGLANG_SIMULATE_ACC_LEN

# The four migrated DSpark operators are loaded from the SGLang tree. The
# installed sgl_kernel_npu package is still used by unrelated Ascend kernels.
export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"

D_IP=('192.168.25.213' '192.168.25.214' '192.168.25.215' '192.168.25.218')
LOCAL_IPS=" $(hostname -I) "
NODE_RANK=-1
for i in "${!D_IP[@]}"; do
    if [[ "$LOCAL_IPS" == *" ${D_IP[$i]} "* ]]; then
        NODE_RANK=$i
        break
    fi
done

if (( NODE_RANK < 0 )); then
    echo "Current host IPs do not match the configured 4-node topology: ${D_IP[*]}" >&2
    exit 2
fi

for config_path in "${MODEL_PATH}/config.json" "${DRAFT_MODEL_PATH}/config.json"; do
    if [[ ! -f "$config_path" ]]; then
        echo "Missing model config on node ${NODE_RANK}: ${config_path}" >&2
        exit 2
    fi
done

GRAPH_ARGS=(
    --cuda-graph-backend-decode disabled
    --cuda-graph-backend-prefill disabled
)
if [[ "$ENABLE_NPU_GRAPH" == "1" ]]; then
    GRAPH_ARGS=()
fi

RADIX_CACHE_ARGS=(--disable-radix-cache)
if [[ "$ENABLE_RADIX_CACHE" == "1" ]]; then
    RADIX_CACHE_ARGS=()
fi

export HCCL_SOCKET_IFNAME=$COMM_IF
export GLOO_SOCKET_IFNAME=$COMM_IF

mkdir -p "${SGLANG_ROOT}/logs"
cd "$SGLANG_ROOT"
LOG_PATH="logs/dspark_32p_node${NODE_RANK}_$(date '+%Y-%m-%dT%H-%M-%S-%3N').log"

echo "K3 DSpark/MTP 4-node launch"
echo "node_rank=${NODE_RANK}, local_ips=${LOCAL_IPS}"
echo "target=${MODEL_PATH}"
echo "draft=${DRAFT_MODEL_PATH}"
echo "dist_init=${MASTER_ADDR}:${MASTER_PORT}, tp=64, dp=4"
echo "npu_graph=${ENABLE_NPU_GRAPH}, radix_cache=${ENABLE_RADIX_CACHE}"
echo "log=${LOG_PATH}"

# K3's MTP implementation is registered as DSPARK. With DP attention, DSpark
# requires the built-in TP MoE path, so moe-a2a-backend must remain "none".
sglang serve \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --dist-init-addr "${MASTER_ADDR}:${MASTER_PORT}" \
    --nnodes 4 \
    --node-rank "$NODE_RANK" \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 64 \
    --enable-dp-attention \
    --dp-size 4 \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --page-size 128 \
    --chunked-prefill-size 8192 \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
    --speculative-dspark-block-size 7 \
    --speculative-draft-attention-backend ascend \
    --speculative-eagle-topk 1 \
    --speculative-draft-model-quantization unquant \
    --speculative-moe-a2a-backend none \
    --enable-multimodal \
    --mm-enable-dp-encoder \
    --mm-attention-backend ascend_attn \
    "${GRAPH_ARGS[@]}" \
    "${RADIX_CACHE_ARGS[@]}" \
    2>&1 | tee "$LOG_PATH"
