export ASCEND_MF_STORE_URL="tcp://127.0.0.1:31001"
export ASCEND_MF_LOG_LEVEL=1
export PYTHONPATH=/home/z00937177/sglang/python:$PYTHONPATH
ASCEND_RT_VISIBLE_DEVICES=2,3 \
sglang serve \
   --model-path /home/weights/Qwen3.6-35B-A3B-w8a8 \
   --served-model-name qwen \
   --disaggregation-mode prefill \
   --tp 2 --dp 1 \
   --host 127.0.0.1 \
   --port 30000 \
   --trust-remote-code \
   --disaggregation-bootstrap-port 8996 \
   --disaggregation-transfer-backend ascend \
   --schedule-policy fcfs \
   --chunked-prefill-size -1 \
   --schedule-conservativeness 0.3 \
   --disable-overlap-schedule \
   --mem-fraction-static 0.4 \
   --max-running-requests 48 \
   --cuda-graph-bs 8 16 24 32 48 \
   --enable-metrics \
   --attention-backend ascend \
   --enable-hierarchical-cache \
   --hicache-io-backend kernel_ascend 2>&1 | tee /home/z00937177/logs/sglang-prefill.log 
   # --hicache-storage-backend ascend_memcache \
   # --hicache-storage-backend-extra-config '{"meta_service_url":"tcp://127.0.0.1:5000", "config_store_url":"tcp://127.0.0.1:6000", "log_level":"info", "world_size":256, "protocol": "device_sdma", "dram_size": "1GB"}'