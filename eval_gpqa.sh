evalscope eval \
    --model /home/weights/Kimi-K3-w4a8-int-moe \
    --api-url http://127.0.0.1:30000/v1 \
    --api-key EMPTY \
    --work-dir "/home/hanwlax/workspace/progress/kimi_k3/gpqa/result_$(date +%Y-%m-%d_%H-%M-%S)" \
    --no-timestamp \
    --eval-type openai_api \
    --datasets gpqa_diamond \
    --dataset-args '{
      "gpqa_diamond": {
        "local_path": "/home/hanwlax/datasets/gpqa",
        "subset_list": ["gpqa_diamond"],
        "default_subset": "gpqa_diamond"
      }
    }' \
    --generation-config '{
      "max_tokens": 131072,
      "timeout": 10000,
      "temperature": 1.0,
      "top_p": 0.95,
      "extra_body": {
        "reasoning_effort": "max"
      }
    }' \
    --eval-batch-size 32 \
    --seed 42