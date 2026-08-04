python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode  http://127.0.0.1:40000 \
    --host 127.0.0.1 \
    --port 9903 \
    --health-check-interval-secs 3600 \
    --mini-lb \