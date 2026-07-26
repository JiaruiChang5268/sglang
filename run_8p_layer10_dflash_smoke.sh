#!/bin/bash
set -eo pipefail

MODEL_PATH="${MODEL_PATH:-/home/zkk/weights/Kimi-K3-int4-layer10}"
DRAFT_PATH="${DRAFT_PATH:-/home/hanwlax/workspace/sglang/local_checkpoints/k3-dflash-smoke-random}"
PORT="${PORT:-8880}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=1600
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_RT_VISIBLE_DEVICES="12,13,14,15"
export PYTHONPATH=/home/hanwlax/workspace/sglang/python:${PYTHONPATH:-}
    # --disable-radix-cache \

exec sglang serve \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 4 \
    --mem-fraction-static 0.6 \
    --context-length 4096 \
    --max-total-tokens 8192 \
    --max-running-requests 4 \
    --page-size 16 \
    --chunked-prefill-size -1 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_PATH" \
    --speculative-draft-load-format dummy \
    --speculative-draft-model-quantization unquant \
    --speculative-draft-attention-backend ascend \
    --speculative-dflash-block-size 1 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --host 127.0.0.1 \
    --enable-multimodal --mm-enable-dp-encoder --mm-attention-backend ascend_attn \
    --port "$PORT" 2>&1 | tee "logs/dflash_$(date '+%Y-%m-%dT%H-%M-%S-%3N').log"