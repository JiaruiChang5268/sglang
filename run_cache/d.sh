export ASCEND_MF_STORE_URL="tcp://127.0.0.1:31001"
export ASCEND_MF_LOG_LEVEL=1
export PYTHONPATH=/home/z00937177/sglang/python:$PYTHONPATH
ASCEND_RT_VISIBLE_DEVICES=10,11 \
sglang serve \
    --model-path /home/weights/Qwen3.6-35B-A3B-w8a8 \
    --served-model-name qwen \
    --disaggregation-mode decode \
    --tp 2 --dp 1 \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 40000 \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8996 \
    --mem-fraction-static 0.4 \
    --max-running-requests 64 \
    --schedule-policy fcfs \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 36000 \
    --schedule-conservativeness 0.3 \
    --disable-overlap-schedule \
    --enable-metrics \
    # --disaggregation-decode-enable-radix-cache \
    # --hicache-io-backend kernel_ascend \
    # --enable-hierarchical-cache \
    # 2>&1 | tee /home/z00937177/logs/sglang-decode.log
    # --disaggregation-decode-enable-offload-kvcache \
    # --hicache-storage-backend ascend_memcache \
    # --hicache-storage-backend-extra-config '{"meta_service_url":"tcp://127.0.0.1:5000", "config_store_url":"tcp://127.0.0.1:6000", "log_level":"info", "world_size":256, "protocol": "device_sdma", "dram_size": "1GB"}'