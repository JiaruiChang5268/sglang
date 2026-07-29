# Kimi-K3 GPQA / SGLang DSPARK 排障交接

更新时间：2026-07-29  
主目录：`/home/hanwlax/test-code/sglang`（213）  
Git 分支：`0728_dspark`  
基线提交：`0809d9b972714d9b7d5dfd4f8312cb346767b68b`

## 1. 当前结论

本轮已完成 Kimi-K3 reasoning parser、`reasoning_effort=max` 传递、constrained reasoning 多 token 结束标记，以及 Ascend DSPARK top-p/top-k sampling fallback 的修复。

服务目前没有再次出现此前的 parser 初始化错误或 `NoneType` top-p kernel 崩溃。209 的健康检查返回 HTTP 200。

当前 GPQA-Diamond 运行停在 197/198：

- 已评分：197
- 正确：157
- 错误：40
- 当前准确率：`157 / 197 = 79.6954%`
- 答案成功解析：197/197
- 唯一未完成样本：dataset index 147
- 若最后一题正确，最终为 `158 / 198 = 79.80%`
- 若最后一题错误，最终为 `157 / 198 = 79.29%`

因此本轮最终分数已基本锁定在 79%–80%，显著低于官方约 94%。这已经不是答案 parser 漏判造成的：当前 197 条全部提取出了 A/B/C/D。

## 2. 集群与路径

四节点服务：

| node | IP | node rank |
|---|---|---:|
| 209 | `192.168.25.209` | 0 |
| 212 | `192.168.25.212` | 1 |
| 216 | `192.168.25.216` | 2 |
| 217 | `192.168.25.217` | 3 |

公共约定：

- 代码目录：`/home/hanwlax/test-code/sglang`
- 主模型：`/home/weights/Kimi-K3-int4`
- draft 模型：`/home/weights/DSpark-Kimi-K3-yi`
- API：`http://192.168.25.209:30000/v1`
- TP：64
- DP：4
- 推测解码：DSPARK，block size 7
- 设备/attention backend：Ascend NPU

启动脚本：

```text
/home/hanwlax/test-code/sglang/run_32p_mix_dspark.sh
```

评测脚本：

```text
/home/hanwlax/test-code/sglang/eval_gpqa.sh
```

## 3. 本轮 benchmark 配置

`eval_gpqa.sh` 当前使用：

```json
{
  "max_tokens": 131072,
  "timeout": 10000,
  "temperature": 1.0,
  "top_p": 0.95,
  "extra_body": {
    "reasoning_effort": "max"
  }
}
```

其他条件：

- EvalScope：1.9.1
- Dataset：`AI-ModelScope/gpqa_diamond`
- 样本数：198
- split：train
- few-shot：0
- eval batch size：32
- seed：42
- choices 未 shuffle
- parser：服务端 `--reasoning-parser kimi_k3`

不要把 `reasoning_effort=max` 放在 EvalScope TaskConfig 顶层。EvalScope 1.9.1 顶层字段只接受 `low/medium/high`，会产生：

```text
ValidationError: reasoning_effort
Input should be 'low', 'medium' or 'high'
```

正确做法是保持现在的 `generation_config.extra_body.reasoning_effort=max`，让它作为 OpenAI 请求扩展字段发送。

## 4. 已修复问题

### 4.1 Kimi-K3 reasoning parser

新增 `kimi_k3` detector，识别 Kimi-K3 XTML channel：

```text
<|open|>think<|sep|>
<|close|>think<|sep|><|open|>response<|sep|>
<|close|>response<|sep|>
```

支持非流式和流式解析，并处理 marker 被拆分到多个 streaming chunk 的情况。

文件：

```text
python/sglang/srt/parser/reasoning_parser.py
test/registered/unit/parser/test_reasoning_parser.py
```

### 4.2 Kimi-K3 effort 参数映射

OpenAI API 仍接收：

```json
{"reasoning_effort": "max"}
```

但 Kimi-K3 chat template 内部需要 `thinking_effort`。服务端针对 Kimi-K3 将：

```text
reasoning_effort -> thinking_effort
```

其他模型仍使用 `reasoning_effort`，不受影响。

文件：

```text
python/sglang/srt/entrypoints/openai/serving_chat.py
test/registered/unit/entrypoints/openai/test_serving_chat.py
```

### 4.3 constrained reasoning 多 token 结束标记

原实现要求 `think_end_token` 必须编码成恰好一个 token。Kimi-K3 的结束/response 开始 marker：

```text
<|close|>think<|sep|><|open|>response<|sep|>
```

会编码成多个 token，因此服务启动时报：

```text
ValueError: think_end_token ... must encode to exactly one token for constrained reasoning.
```

现已支持多 token marker：

- prefix 增量匹配
- mismatch 后恢复普通 reasoning token 计数
- token filter 只允许 marker 的下一个 token
- copy/rollback 保存完整匹配状态

文件：

```text
python/sglang/srt/constrained/reasoner_grammar_backend.py
test/registered/unit/constrained/test_reasoner_grammar_backend.py
```

### 4.4 Ascend DSPARK top-p/top-k fallback

此前四台服务在第一批 GPQA 请求进入 DSPARK verification 后全部崩溃，核心栈为：

```text
build_dflash_verify_target_probs
top_p_renorm_prob(...)
TypeError: 'NoneType' object is not callable
```

根因：

- CUDA/MUSA 可导入 `sgl_kernel` 的 `top_p_renorm_prob`。
- Ascend/NPU 分支把 `top_p_renorm_prob` 和 `top_k_renorm_prob` 设为 `None`。
- GPQA 设置 `top_p=0.95`，使 `need_top_p_sampling=True`。
- DSPARK 仍无条件调用了 `None`。

正确修复：

- CUDA/MUSA 继续使用原高性能 kernel。
- kernel 不存在时，top-p 使用现有 `top_p_normalize_probs_torch` 精确回退。
- top-k 同时增加 PyTorch `topk + mask + renormalize + scatter` 回退。
- 只在对应 sampling 功能实际启用时进入 fallback。

文件：

```text
python/sglang/srt/speculative/dflash_utils.py
test/registered/unit/speculative/test_dflash_utils.py
```

性能说明：

- `top_p=1.0` 或 greedy 请求不会调用 top-p fallback，几乎没有性能影响。
- 本轮 GPQA 使用 `top_p=0.95`，会调用 PyTorch fallback。
- fallback 的全词表 sort/cumsum/scatter 可能显著慢于 fused kernel，但语义正确。
- 如需量化影响，应对相同请求做 `top_p=1.0`、`top_p=0.95` 和未来 Ascend fused kernel 的 A/B benchmark。

## 5. 跨节点一致性

以下四个运行时文件在 213、209、212、216、217 上 SHA256 完全一致：

```text
644fed2b82c8a53b60c782e97de23bb5f2b17ab504c150bd7dd2ade7c58cfc4b  python/sglang/srt/constrained/reasoner_grammar_backend.py
00f8f8a5b9f054c6aaf2e78d38badf2e0fdd0f345885cc5c975c387eb69a8264  python/sglang/srt/entrypoints/openai/serving_chat.py
18aca83915ff4bd76deccfdd3ce8b1c90c6109220b0cae08b53f389192ce523f  python/sglang/srt/parser/reasoning_parser.py
ec4a6dc710349e861b014bb01805bd40ce8f8a280cf3776d6d179b925a98b66f  python/sglang/srt/speculative/dflash_utils.py
```

DSPARK fallback 测试文件哈希：

```text
4982b0d6a714a74c847bfb7657dead19b0daf1f9741fa023f6a4afb57cb6898f  test/registered/unit/speculative/test_dflash_utils.py
```

## 6. 已完成验证

### 单元测试

已运行并通过：

- constrained reasoning backend：28 tests
- reasoning parser：128 tests
- OpenAI serving chat：73 tests
- DSPARK fallback：2 tests

DSPARK 两个新增测试覆盖：

- `top_p_renorm_prob is None` 时的精确 top-p fallback
- `top_k_renorm_prob is None` 时的精确 top-k fallback

### 真实 NPU smoke test

在 209 的现有 SGLang 容器内运行了真实 Ascend tensor：

```text
kernel_is_none= True
device= npu:0
probs= [[[0.7310585975646973, 0.2689414322376251, 0.0, 0.0]]]
sum= [[1.0]]
```

这证明不是仅在 CPU 单测中通过，Ascend 实际设备上也能执行并正确归一化。

### 服务验证

- 四节点服务能通过 parser/grammar 初始化和 warmup。
- GPQA top-p 请求不再触发 `NoneType` 崩溃。
- 209 `/health` 当前返回 HTTP 200。
- 197 个样本均返回可解析最终答案。

## 7. 当前评测运行

输出目录：

```text
/home/hanwlax/test-code/sglang/outputs/20260728_190646
```

关键文件：

```text
configs/task_config.yaml
logs/eval_log.log
predictions/Kimi-K3-int4/gpqa_diamond_default.jsonl
reviews/Kimi-K3-int4/gpqa_diamond_default.jsonl
```

运行开始时间：

```text
2026-07-28 19:06:46
```

当前状态：

```text
197/198
missing dataset index: 147
evalscope PID on 213: 2179827
```

截至 2026-07-29 00:45 左右，日志仍每分钟打印 197/198，最后一个请求长期未返回；evalscope 进程仍存活，API health 为 200。

本轮 `timeout=10000` 秒且 API client 配置有最多 5 次 retry，单个异常长请求可能阻塞很久。不要因为健康检查正常就认为 index 147 一定仍在有效生成，也可能处于等待、重试或某个 DP rank 的长尾状态。

准确率演变：

```text
前 34 条：32/34 = 94.12%
前 40 条：38/40 = 95.00%
当前 197 条：157/197 = 79.70%
```

早期小样本明显不代表最终结果。

旧输出目录：

```text
outputs/20260728_184720
```

该轮在 DSPARK top-p `NoneType` 崩溃后没有有效 predictions，不应纳入评分。

`outputs/20260728_161849` 使用旧代码/旧 benchmark 配置及不同 model id，也不应与当前轮直接合并。

## 8. 为什么仍低于官方 94%

当前可以排除：

- 不是服务启动失败：服务健康且已完成 197 条。
- 不是 Kimi-K3 channel parser 全面失效：197/197 答案已解析。
- 不是 EvalScope 把 `max` 拒绝：`max` 已通过 `extra_body` 发送。
- 不是 DSPARK top-p `NoneType` 崩溃：该问题已修复。

仍需优先验证：

1. **模型/量化版本是否与官方分数一致**  
   当前是 `/home/weights/Kimi-K3-int4`。官方 94% 可能来自不同 checkpoint、精度、量化方式或服务实现。

2. **官方评测 recipe 是否完全一致**  
   需要逐项确认官方 dataset revision、prompt、choice 顺序、0/5-shot、temperature/top-p、seed、是否多次采样或 majority vote。

3. **DSPARK 路径是否影响答案质量**  
   需要对相同题号进行 DSPARK on/off A/B。正确的 sampling fallback 只解决崩溃，不证明 draft/verify 的整体数值行为与无 speculative decoding 完全一致。

4. **reasoning effort 是否真的在模板中生效**  
   代码已做 `reasoning_effort -> thinking_effort` 映射，输出也包含 reasoning；仍建议抓一条最终渲染后的 chat template 或请求 trace，确认 `max` 没被模型模板忽略。

5. **随机单次评测波动**  
   本轮 `temperature=1.0, top_p=0.95`，单次 198 题存在随机性，但从约 94% 降至约 80% 通常不应仅归因于普通采样波动。

建议第一优先级是：固定一批错题，分别运行“当前 DSPARK”和“关闭 speculative decoding”的相同参数 A/B，再对比最终答案和 reasoning。

## 9. 工作区状态

213 工作区目前有未提交修改，不要执行 `git reset --hard` 或覆盖式 checkout。

修改文件：

```text
python/sglang/srt/constrained/reasoner_grammar_backend.py
python/sglang/srt/entrypoints/openai/serving_chat.py
python/sglang/srt/parser/reasoning_parser.py
python/sglang/srt/speculative/dflash_utils.py
run_32p_mix_dspark.sh
test/registered/unit/constrained/test_reasoner_grammar_backend.py
test/registered/unit/entrypoints/openai/test_serving_chat.py
test/registered/unit/parser/test_reasoning_parser.py
```

未跟踪文件：

```text
eval_gpqa.sh
test/registered/unit/speculative/test_dflash_utils.py
```

`run_32p_mix_dspark.sh` 在主节点匹配循环结束后有 `exit 1`，其后的示例 `sglang server`、`bench_serving`、`curl` 块均不可达。提交前应删除或移到独立示例脚本，避免误导；不要在当前评测运行中修改并重启服务。

## 10. 常用检查命令

进入目录：

```bash
ssh 213
cd /home/hanwlax/test-code/sglang
```

查看当前结果数量：

```bash
wc -l outputs/20260728_190646/{predictions,reviews}/Kimi-K3-int4/gpqa_diamond_default.jsonl
```

跟踪评测：

```bash
tail -f outputs/20260728_190646/logs/eval_log.log
```

检查 evalscope 进程：

```bash
pgrep -af "evalscope eval"
```

检查 API：

```bash
ssh 209 'curl -sS -o /dev/null -w "%{http_code}\n" --max-time 5 http://127.0.0.1:30000/health'
```

统计当前准确率：

```bash
python3 -c 'import json; p="outputs/20260728_190646/reviews/Kimi-K3-int4/gpqa_diamond_default.jsonl"; xs=[json.loads(l) for l in open(p) if l.strip()]; s=[x["sample_score"]["score"]["value"]["acc"] for x in xs]; print(len(s), sum(s), sum(s)/len(s))'
```

查看改动：

```bash
git status --short
git diff --check
git diff --stat
```

## 11. 建议接手顺序

1. 先确认 index 147 是否最终返回，以及结果目录是否变成 198 行。
2. 若仍卡住，保存日志并定位该请求所在 DP rank；不要直接删除当前输出。
3. 汇总完整 198 条结果和错题题号。
4. 选 10–20 道错题做 DSPARK on/off、int4/更高精度模型 A/B。
5. 与官方 Kimi-K3 GPQA recipe 逐项对齐。
6. 清理 `run_32p_mix_dspark.sh` 的不可达调试块。
7. 对全部 diff 做 review 后再提交；当前修改尚未 commit。

