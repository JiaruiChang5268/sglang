evalscope eval \
    --model /home/weights/Kimi-K3-int4 \
    --api-url http://192.168.25.209:30000/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --generation-config '{
      "max_tokens": 131072,
      "timeout": 10000,
      "temperature": 1.0,
      "top_p": 0.95,
      "extra_body": {
        "reasoning_effort": "max"
      }}' \
    --datasets gpqa_diamond \
    --eval-batch-size 32
