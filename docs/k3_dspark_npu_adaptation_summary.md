# K3 适配 DSpark 与 NPU 支持修改总结

## 1. 文档范围

本文总结 Kimi K3 在 SGLang 中适配 DSpark speculative decoding 的主要修改、运行逻辑、NPU/KDA 状态管理方案，以及当前 `sgl-kernel-npu` 的修改边界。

分析基于远端代码：

```text
Host: 213-kimi-hw
Repository: /home/hanwlax/workspace/sglang
Branch: 0721_zkk
DSpark commit: 8b1b235ab43f96fec4d8eb3f1fafd889472e3861
Base commit: 66034bff8
```

最终验证状态（2026-07-25）：

```text
Prefix cache: enabled and hit
Target NPU graph: enabled, bs=[1..8], 8 verify tokens/request
Draft NPU graph: enabled, bs=[4,8], 7 draft tokens/request
Acceptance simulation: disabled
Real accepted length: 1.015873
Correct draft tokens: 0
Final log: logs/dspark_2026-07-25T16-23-28-871.log
```

提交规模：

```text
69 files changed
13354 insertions
257 deletions
```

这次适配不是只增加一个 DSpark 模型类，而是包含以下部分的一整套改动：

1. DSpark 算法注册和启动参数。
2. DSpark draft 模型及 Markov head。
3. K3 Target 中间层 hidden capture。
4. Target hidden 到 Draft KV 的注入。
5. Draft proposal、Target Verify 和 acceptance。
6. K3 KDA/Mamba speculative state 的保存、提交和回滚。
7. Ragged Verify、CUDA/NPU Graph 和 overlap scheduler 支持。
8. Acceptance、cap 和 SPS 等指标输出。
9. NPU Torch fallback。

---

## 2. 总体执行链路

```text
请求 Prefill
  ↓
K3 Target 正常前向
  ↓
捕获指定 Target 层的有效 residual stream
  ↓
拼接 aux hidden
  ↓
DSpark fc 投影
  ↓
生成并写入 Draft 模型 KV

Decode
  ↓
上轮 bonus token + mask token
  ↓
DSpark Draft backbone 一次并行前向
  ↓
Target lm_head 生成 base logits
  ↓
Markov head 左到右修正并采样 γ 个 draft token
  ↓
可选 confidence head 预测每个位置的成功率
  ↓
Verify Planner 决定每个请求验证多少 token
  ↓
K3 Target 验证 anchor + γ drafts
  ↓
Greedy/Sampling acceptance
  ↓
提交接受 token 的：
  ├─ Target token/KV
  ├─ Draft hidden/KV
  └─ KDA conv/SSM state
  ↓
bonus token 进入下一轮
```

核心长度关系：

```text
gamma = DSpark 提出的 draft token 数
verify_num_draft_tokens = gamma + 1
```

多出来的一个位置用于 anchor/bonus token。

---

## 3. 算法注册与启动参数

主要文件：

```text
python/sglang/srt/speculative/spec_info.py
python/sglang/srt/speculative/spec_registry.py
python/sglang/srt/arg_groups/speculative_hook.py
python/sglang/srt/server_args.py
python/sglang/srt/environ.py
```

### 3.1 注册 DSPARK

在 `SpeculativeAlgorithm` 中增加：

```python
DSPARK = auto()
```

同时增加能力判断：

```python
def is_dspark(self):
    return self == SpeculativeAlgorithm.DSPARK

def is_dflash_family(self):
    return self.is_dflash() or self.is_dspark()

def supports_ragged_verify(self):
    return self.is_dspark()
```

这里形成了两层关系：

- DSpark 和 DFLASH 共用 block draft、Target hidden capture、Draft KV、Target Verify 等基础设施。
- Ragged Verify 当前仅由 DSpark 支持。
- DSpark 并不是 DFLASH 的别名。

### 3.2 DSpark 启动校验

`_handle_dspark()` 负责：

- 限制运行设备为 CUDA 或 NPU。
- 当前限制 `pp_size == 1`。
- 强制 `speculative_num_steps == 1`。
- 强制 `speculative_eagle_topk == 1`。
- 校验 `speculative_num_draft_tokens == gamma + 1`。
- 禁用 mixed chunked prefill。
- 校验 DP attention、DP lm_head、MoE A2A 和 CP 组合。
- 支持从 Target checkpoint 中识别捆绑的 `dspark_*` draft 权重。

### 3.3 新增参数

```text
--speculative-dspark-block-size
--speculative-dspark-sps-table-path
--speculative-dspark-confidence-sts-path
--speculative-dspark-align-verify-tokens-to-graph-tier
```

### 3.4 新增环境变量

```text
SGLANG_RAGGED_VERIFY_MODE
SGLANG_DSPARK_FAST_SAMPLING
SGLANG_DSPARK_FAST_KERNEL
SGLANG_DSPARK_ENABLE_MULTI_STREAM
SGLANG_DSPARK_STS_COLLECT_PATH
SGLANG_DSPARK_ENABLE_SPS_RECORD
SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL
SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH
```

---

## 4. DSpark 配置和权重解析

主要文件：

```text
python/sglang/srt/speculative/dspark_components/dspark_config.py
```

DSpark 在 DFLASH 配置基础上扩展：

```text
block_size / gamma
markov_rank
markov_head_type
mask_token_id
target_layer_ids
num_target_layers
num_hidden_layers
```

支持嵌套配置：

```json
{
  "dspark_config": {
    "markov_rank": 128,
    "markov_head_type": "vanilla"
  }
}
```

也支持 checkpoint 中的扁平字段：

```json
{
  "dspark_block_size": 7,
  "dspark_markov_rank": 128,
  "dspark_markov_head_type": "vanilla",
  "dspark_noise_token_id": 163839,
  "dspark_target_layer_ids": [7, 23, 51, 67, 83]
}
```

配置解析优先级：

```text
dspark_* 扁平字段
  > dspark_config
  > text_config
  > DFLASH 公共配置
```

必要条件：

- `markov_rank > 0`。
- `mask_token_id` 存在且小于 Target vocabulary size。
- `gamma >= 1`。
- `target_layer_ids` 非空且位于 Target 层范围内。

---

## 5. K3 Target hidden capture

主要文件：

```text
python/sglang/srt/models/kimi_k3.py
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/layers/logits_processor.py
```

### 5.1 设置捕获层

K3 模型新增：

```python
set_dspark_layers_to_capture(layer_ids)
```

`ModelRunner` 从 Draft config 解析 `target_layer_ids`，随后调用：

```python
self.model.set_dspark_layers_to_capture(target_layer_ids)
```

当前实现要求：

```text
PP size = 1
```

### 5.2 捕获有效 residual stream

K3 在指定层执行：

```python
aux_hidden_states.append(
    self._dspark_capture_stream(i, hidden_states, residual)
)
```

不能直接使用裸 `hidden_states`，因为 K3 的层输出包含：

```text
hidden_states
residual
```

下一层实际消费的是经过 residual 聚合、投影和 norm 的 stream。

`_dspark_capture_stream()` 的逻辑是：

```text
未启用 attention residual:
    hidden_states 或 hidden_states + residual

启用 attention residual 且不是最后一层:
    使用下一层的 self_attention_res_proj/res_norm

启用 attention residual 且是最后一层:
    使用 output_attn_res_proj/output_attn_res_norm
```

这样得到的是“当前层之后、下一层之前”的有效语义流。

`_dspark_capture_stream()` 被 DFLASH family 公共路径复用并不表示 DFLASH
执行了 DSpark 算法。它只负责从 K3 层内的 `hidden_states + residual`
表示中还原可供 drafter 学习/消费的 target stream；DSpark 与 DFLASH
在这一阶段的数据语义相同。函数名来自最初的 DSpark 落地，后续可重命名为
更中性的 `capture_drafter_target_stream()`，但当前保留名称可避免扩大改动。

### 5.3 拼接 aux hidden

`LogitsProcessor` 将多个捕获层沿最后一维拼接：

```python
aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
```

例如捕获 5 层、hidden size 为 7168：

```text
[N, 7168] × 5
    ↓ concat
[N, 35840]
```

---

## 6. DSpark Draft 模型

主要文件：

```text
python/sglang/srt/models/dspark.py
python/sglang/srt/models/dflash.py
```

模型继承关系：

```python
class DSparkDraftModel(DSparkDraftMixin, DFlashDraftModel):
    pass
```

含义：

- 复用 DFLASH Draft backbone。
- 增加 DSpark Markov head。
- 可选增加 confidence head。
- 复用 Target hidden 到 Draft KV 的投影和写入逻辑。

Target hidden 的处理链路：

```text
concat Target hidden
  ↓
fc: K × hidden_size → hidden_size
  ↓
hidden_norm
  ↓
每层 kv_proj_only
  ↓
写入 Draft KV cache
```

DSpark Draft 不加载独立 embedding 和 lm_head：

```python
_DSPARK_SKIPPED_WEIGHT_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "rotary_emb.",
)
```

运行时共享 Target 的：

```text
Target embedding
Target lm_head
```

这样既减小 Draft checkpoint，也保证 vocabulary projection 与 Target 对齐。

---

## 7. Markov head 与候选生成

支持三种 Markov head：

```text
vanilla
gated
rnn
```

### 7.1 Vanilla Markov

```text
previous token
  ↓ embedding(markov_rank)
  ↓ linear → vocabulary
  ↓
base logits + Markov bias
```

### 7.2 Gated Markov

```text
draft hidden + previous-token embedding
  ↓ sigmoid gate
  ↓ gated Markov projection
```

### 7.3 RNN Markov

在一个 Draft block 内维护 Markov state，让前一个 token 影响后续 proposal。

### 7.4 Proposal 过程

`DraftBlockProposer` 构造长度为 `gamma` 的 Draft 输入：

```text
[bonus_token, mask_token, mask_token, ...]
```

Draft backbone 一次前向产生：

```text
draft_hidden: [batch_size, gamma, hidden_size]
```

共享 Target lm_head 产生：

```text
base_logits: [batch_size, gamma, vocabulary_size]
```

Markov head 从左到右修正并采样：

```text
token_1 → 修正 token_2 logits
token_2 → 修正 token_3 logits
...
```

最终 Target Verify 输入为：

```text
[anchor/bonus, draft_1, draft_2, ..., draft_gamma]
```

---

## 8. Prefill 与 Draft KV 初始化

主要入口：

```text
DSparkWorkerV2._forward_prefill()
```

执行过程：

1. 将 Target 的 `capture_hidden_mode` 设置为 `FULL`。
2. K3 Target 完成正常 Prefill。
3. 从 `LogitsProcessorOutput` 取得拼接后的 aux hidden。
4. 根据 `prefix_lens` 和 `extend_lens` 计算 position。
5. 调用 `TargetHiddenKvInjector.inject_target_hidden()`。
6. 将 Target hidden 投影为 Draft hidden。
7. 为 Draft 每一层生成 K/V 并写入 Draft KV cache。
8. Target 生成的 next token 成为下一轮 bonus token。

Draft 不会从空上下文开始，而是通过 Target hidden 投影获得与 Target 上下文对齐的 KV。

---

## 9. Decode、Target Verify 与 Acceptance

主要入口：

```text
DSparkWorkerV2._forward_decode()
```

### 9.1 Verify window

为每个请求分配：

```text
positions
Target KV location
Draft KV location
```

固定模式下的形状：

```text
[batch_size, gamma + 1]
```

### 9.2 Draft proposal

`DraftBlockProposer.propose()` 完成：

- bonus + mask 输入构造。
- Draft backbone 前向。
- Target lm_head 生成 base logits。
- Markov head 生成 Draft token。
- 可选计算 confidence。

### 9.3 Verify Planner

支持：

```text
static
cap-accept
compact
```

#### static

- 不构造 confidence head。
- 不加载 SPS table。
- 每个请求验证完整 `gamma + 1`。
- 不创建 `RaggedVerifyLayout`。
- 不做 compact token packing。

#### cap-accept / compact

- 使用 confidence 估计各位置成功率。
- 使用 SPS 表估算不同 token 数的执行成本。
- 为每个请求生成不同的 `verify_lens`。
- compact 模式按实际 token 总数选择 Graph tier。

### 9.4 Target Verify

Target 对以下序列执行一次前向：

```text
anchor + gamma drafts
```

同时捕获每个验证位置对应的 Target hidden。

### 9.5 Acceptance

Greedy 模式：

- 从第一个 Draft token 开始与 Target argmax 比较。
- 连续接受，直到第一个不匹配位置。
- 返回 `correct_len` 和 bonus token。

Sampling 模式：

- 使用 Target 概率和 Draft 概率执行无损 speculative sampling。
- 支持同一个 batch 中混合 greedy 和 sampling 请求。

最终：

```text
commit_lens = correct_len + 1
```

其中 `+1` 是必须提交的 bonus/anchor token。

---

## 10. Draft hidden/KV 提交

Target Verify 会生成完整 Verify block 的 Target hidden，但不能将全部位置写入持久 Draft KV。

`TargetHiddenKvInjector` 根据每个请求的 `commit_lens`：

```text
选取前 commit_lens 个 Target hidden
  ↓
fc 投影
  ↓
生成 Draft K/V
  ↓
写入持久 Draft KV
```

被拒绝部分仅存在于 speculative scratch，不进入持久缓存。

---

## 11. K3 KDA/Mamba 状态适配

主要文件：

```text
python/sglang/srt/layers/attention/linear/kda_backend.py
python/sglang/srt/layers/attention/linear/kernels/kda_triton.py
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py
python/sglang/srt/hardware_backend/npu/memory_pool_npu.py
python/sglang/srt/mem_cache/memory_pool.py
```

### 11.1 KDA conv cache 维度

普通 Mamba 与 KDA 的 conv state 逻辑布局不同：

```text
普通 Mamba: [channels, window]
KDA:        [window, channels]
```

之前公共 NPU 初始化逻辑直接给 conv window 增加 speculative 长度，曾导致：

```text
实际维度：9217
Q/K/V split 期望总和：9216
```

修复方式：

- 通过 `is_kda` 区分 KDA 和普通 Mamba。
- KDA 的持久 conv state 保持原始 window 长度。
- 每个 speculative step 的状态写入 intermediate scratch。
- 不再把 Draft 长度加入 KDA 基础 conv state。

### 11.2 NPU causal-conv Target Verify

在 SGLang 中增加：

```python
_npu_causal_conv1d_linear_verify()
```

按 token 顺序执行：

```text
旧 conv state + 当前 token
  ↓
计算 conv 输出
  ↓
窗口左移
  ↓
保存当前 token 后的 conv snapshot
```

每一步都保存 snapshot，Acceptance 后可精确提交最后一个接受位置的状态。

### 11.3 KDA SSM state

增加：

```python
kda_target_verify_torch_native()
```

最初的 NPU fallback 通过 `cu_seqlens.cpu().tolist()` 和
`Tensor.item()` 遍历请求，这会在 NPU graph capture 中触发 captured-stream
同步错误。最终实现利用 DSpark `static` verify 的固定宽度：

1. 用 `cache_steps` 将 Q/K/V/A/B reshape 为
   `[batch, verify_width, ...]`。
2. 通过 device-side `index_select` 一次取出所有请求的初始 SSM state。
3. 只保留固定 `verify_width` 的 Python 循环，不读取任何 device scalar。
4. 每一步批量执行 KDA recurrence 并保存 SSM snapshot。
5. Verify 阶段不直接覆盖持久 SSM state。

同时修复 K3 GQA：

```text
Q head number
K head number
V head number
```

分别计算 repeat ratio，不再假设 Q/K/V head 数相同。

随机小张量与旧逐请求 reference 的数值对比：

```text
output max abs diff: 2.9802322387695312e-08
scratch max abs diff: 1.1920928955078125e-07
```

### 11.4 状态提交与回滚

Target Verify 完成后调用：

```python
commit_mamba_states_after_verify(...)
```

根据 `commit_lens`：

- 找到每个请求最后一个接受位置。
- 将对应 intermediate SSM snapshot 写回持久 state。
- 将对应 conv snapshot 写回持久 conv cache。
- 丢弃所有被拒绝位置的状态。

Ascend 路径增加：

```python
_move_intermediate_cache_torch()
_conv_state_rollback_torch()
```

用于绕开现有 NPU Triton rollback/move kernel 在 K3 大 state 下可能产生的超大 UB tile。

最终又增加了一项 DSpark/KDA 专用修复：

- KDA verify 已保存每一步 conv window，布局为
  `[layer, request, step, channel, window]`。
- Acceptance 后直接按 `last_correct_step_indices` 将对应 snapshot 写回
  `[layer, slot, channel, window]`。
- Prefix cache tracking slot 同样从准确的 step snapshot 提交。
- 不再用“最终窗口向右平移”推算早期状态；当 draft 数大于 conv window
  时，该旧算法无法恢复已被覆盖的历史。
- 修改只在 `linear_attn_backend._dspark_target_verify` 为真时启用；
  GDN 等旧 NPU backend 继续使用原 rollback 逻辑。

---

## 12. Ascend Attention metadata

主要文件：

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py
```

Target Verify 的 block-table 宽度统一从以下数据计算：

```python
forward_batch.seq_lens_cpu.max().item()
    + speculative_num_draft_tokens
```

原因：

- Overlap scheduler 可能已经将 CPU sequence length 发布到下一步。
- Device `seq_lens` 仍可能保留上一步状态。
- 混用两套长度在 page boundary 上可能导致 block table 少一页或少一个 token。

修复后 block table 与 FIA 实际使用的 `seq_lens_cpu` 保持一致。

---

## 13. Ragged Verify 与 Graph

主要文件：

```text
python/sglang/srt/speculative/ragged_verify.py
python/sglang/srt/speculative/ragged_verify_kernels.py
python/sglang/srt/speculative/dspark_components/dspark_planner.py
python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
```

`RaggedVerifyLayout` 保存：

```text
verify_lens
qo_indptr_device
total_verify_tokens
graph_num_tokens
extend_start_loc
max_q_len
max_kv_len
```

传统固定 Verify Graph 主要按 batch size 选图：

```text
graph key = batch_size
```

Ragged compact 后，同一个 batch size 可能对应不同 token 数，因此修改为：

```text
graph key = graph_num_tokens
```

并让 FlashAttention、FlashInfer、TRTLLM、Hybrid Linear Attention 和 KDA backend 理解 ragged query indptr 和 per-request verify length。

最终 NPU 验证使用：

```bash
SGLANG_RAGGED_VERIFY_MODE=static
--cuda-graph-max-bs-decode 8
```

Target verify graph 和 Draft graph 都已实际捕获并 replay。`static` 模式下：

```text
Target graph: num_tokens_per_bs=8, bs=[1,2,3,4,5,6,7,8]
Draft graph:  num_tokens_per_bs=7, bs=[4,8]
Decode log:   npu graph: True
```

`--cuda-graph-max-bs-decode 8` 是必要的容量约束：Target verify 每个请求
固定 8 tokens，最大 batch 8 对应 64 tokens，恰好不超过
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64`。

为把 DSpark graph 尾部工作纳入公共 Decode graph，在
`ModelRunner` 增加 `capture_tail_hooks`，并在
`DecodeCudaGraphRunner` 完成模型前向后逐个调用。DSpark 用该 hook 将
verify epilogue 或 draft greedy proposal 折叠进 graph；原 DFLASH sampler
路径保持不变。

---

## 14. Overlap scheduler 与 confidence relay

主要文件：

```text
python/sglang/srt/managers/overlap_utils.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/mem_cache/memory_pool.py
```

动态 Ragged Verify 需要让下一轮 scheduler 得到上一轮 confidence。

新增 `ConfidenceRelay`：

```text
GPU confidence buffer
  ↓ asynchronous D2H
pinned CPU ring buffer
  ↓ fixed lag
下一轮 Verify budget planner
```

Request pool slot 还增加：

```python
req_generation
```

因为 request slot 会复用；generation 用于阻止新请求读到旧请求残留的 confidence。

`static` 模式不会启用 confidence relay。

---

## 15. 可观测性与结果输出

主要文件：

```text
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/managers/scheduler_components/batch_result_processor.py
python/sglang/srt/managers/scheduler_components/output_streamer.py
python/sglang/srt/managers/tokenizer_manager.py
python/sglang/srt/managers/detokenizer_manager.py
python/sglang/srt/managers/io_struct.py
python/sglang/srt/observability/metrics_collector.py
```

新增统计字段：

```text
spec_num_block_accept_tokens
spec_num_cap_tokens
spec_cap_lens_histogram
spec_block_accept_length
spec_cap_length
```

同时增加：

- SPS profiler。
- STS calibration fitter。
- Block acceptance estimator。
- DSpark step dump 和动态调试控制。

---

## 16. 为什么修改公共路径

### 16.1 DFLASH family 公共路径

```text
model_runner.py
pool_configurator.py
dflash_info.py
dflash_info_v2.py
dflash_utils.py
draft_worker_common.py
```

DSpark 和 DFLASH 都需要：

- 独立 Draft worker。
- Target hidden capture。
- Draft KV cache。
- Block Target Verify。
- Bonus token 跨轮传递。
- Target/Draft 共享 embedding 或 lm_head。

因此抽象为 `is_dflash_family()`，避免复制整套 runtime。

### 16.2 Attention/Graph 公共路径

```text
base_attn_backend.py
flashattention_backend.py
flashinfer_backend.py
trtllm_mha_backend.py
decode_cuda_graph_runner.py
hybrid_linear_attn_backend.py
```

Ragged Verify 会改变：

```text
query indptr
token count
graph key
KV length
cache location
```

这些必须由公共 attention/graph 层处理，不能仅在 DSpark worker 内处理。

### 16.3 Scheduler/输出公共路径

需要跨 scheduler、worker、detokenizer 和 HTTP 输出传递：

```text
accept_lens
block_accept_lens
cap_lens
confidence
verify token budget
```

因此需要修改公共数据结构和结果处理路径。

---

## 17. `sgl-kernel-npu` 是否有修改

结论：

> 当前 K3 DSpark 适配没有修改 `sgl-kernel-npu` 的 tracked 源码，NPU speculative 兼容逻辑全部实现在 SGLang Python/Torch fallback 中。

核对的源码仓库：

```text
Repository: /home/hanwlax/workspace/sgl-kernel-npu
Branch: situ-quant
HEAD: a4ee5fe29f1e43dbc678b446924a11ae7cda6399
```

工作区仅有以下无关未跟踪内容：

```text
?? csrc/deepep/ops/CMakePresets.json
?? csrc/deepep/ops/third_party/
```

运行环境安装包：

```text
Package: sgl_kernel_npu
Version: 2026.6.1
Location: /usr/local/python3.11.15/lib/python3.11/site-packages
```

### 17.1 当前职责边界

```text
sgl-kernel-npu
  ├─ 提供普通 NPU causal-conv 等已有算子
  └─ 没有新增 DSpark 专用算子

SGLang
  ├─ DSpark linear-chain conv verify
  ├─ KDA Target Verify
  ├─ speculative state snapshots
  ├─ accepted-state commit
  └─ rejected-state rollback
```

### 17.2 当前 Torch fallback

SGLang 中新增或替代：

```text
_npu_causal_conv1d_linear_verify
kda_target_verify_torch_native
_move_intermediate_cache_torch
_conv_state_rollback_torch
```

这些实现优先保证正确性，但包含：

- Python 循环。
- 逐 token 执行。
- 额外 state copy。

其中 KDA target verify 的固定循环已可被 NPU graph capture，不再调用
`.cpu()`/`.item()`；`_move_intermediate_cache_torch` 仍在 graph replay
之外按请求使用 `.item()`，因此仍是性能 fallback。

### 17.3 后续建议下沉到 `sgl-kernel-npu`

建议实现：

1. KDA linear-chain causal-conv verify，并输出每一步 conv snapshot。
2. KDA Target Verify，并输出每一步 SSM snapshot。
3. 批量 `move_intermediate_cache`。
4. 按每个请求 accepted step 批量执行 conv state commit/rollback。
5. 后续按需要增加 Ragged Verify 版本。

SGLang 当前 Torch 实现可以继续作为 correctness fallback。

明确的算子 TODO：

| 优先级 | 待实现 NPU 算子 | 当前 fallback | 目的 |
|---|---|---|---|
| P0 | Batched accepted-state scatter | `_move_intermediate_cache_torch` | 去掉逐请求 `.item()` 和 host/device 同步 |
| P1 | KDA target-verify recurrence + per-step SSM snapshots | `kda_target_verify_torch_native` | 去掉固定 token 循环并提升 graph replay 性能 |
| P1 | KDA linear-chain causal-conv + per-step window snapshots | `_npu_causal_conv1d_linear_verify` | 融合 conv、激活和 snapshot 写入 |
| P2 | 通用 conv accepted-step commit/rollback | `_conv_state_rollback_torch`（非 DSpark legacy path） | 加速 GDN/其他 Mamba speculative backend |

本轮没有使用“只返回正确 shape”的 mock 值：所有验证均执行真实 target、
draft、KDA recurrence、conv recurrence 和 acceptance。由于 Torch fallback
已能正确跑通，未修改 `sgl-kernel-npu` tracked 源码。

---

## 18. 当前 NPU smoke 配置

当前脚本：

```text
/home/hanwlax/workspace/sglang/run_8p_layer10.sh
```

关键配置：

```bash
MODEL_PATH=/home/zkk/weights/Kimi-K3-int4-layer10
DRAFT_MODEL_PATH=/home/hanwlax/workspace/checkpoints/DSpark-Kimi-K3-layer6-smoke

export SGLANG_RAGGED_VERIFY_MODE=static
unset SGLANG_SIMULATE_ACC_LEN

--tp-size 4
--speculative-algorithm DSPARK
--speculative-dspark-block-size 7
--speculative-draft-attention-backend ascend
--cuda-graph-max-bs-decode 8
--chunked-prefill-size 2048
```

长度关系：

```text
gamma = 7
verify width = 8
```

Prefix cache 和 target/draft decode graph 均开启。没有设置
`SGLANG_SIMULATE_ACC_LEN`，所有 acceptance 指标均来自真实 draft/target
token 比较。

---

## 19. 当前已覆盖和未覆盖范围

### 19.1 已覆盖

- K3 Target 裁层模型加载。
- DSpark Draft 裁层模型加载。
- K3 Target aux hidden capture。
- Target hidden 到 Draft KV 的投影和注入。
- Draft proposal 前向。
- Target Verify。
- KDA conv/SSM speculative scratch。
- Acceptance 后按 accepted step 精确提交 KDA conv/SSM state。
- 8 并发共享前缀请求。
- 7169-token prompt + 512-token completion 长序列。
- Prefix cache 命中和 Mamba tracking。
- Target 与 Draft NPU graph capture/replay。
- `static` 模式下模拟 accepted length=4 的多步提交验证。
- 关闭模拟后的真实 acceptance。

最终真实重复前缀验证：

```text
request 1:
  prompt tokens: 2816
  cached tokens: 0
  completion tokens: 64

request 2:
  prompt tokens: 2816
  cached tokens: 2688
  completion tokens: 64

both:
  spec_accept_length: 1.0158730158730158
  spec_accept_rate: 0
  spec_num_correct_drafts: 0
  decode npu graph: True
```

### 19.2 重要限制

裁层权重只适合验证 runtime/shape/state，不适合验证 DSpark 质量：

```text
Target:
  path: /home/zkk/weights/Kimi-K3-int4-layer10
  actual text layers: 6
  hidden size: 7168
  vocab size: 163840

Draft:
  path: /home/hanwlax/workspace/checkpoints/DSpark-Kimi-K3-layer6-smoke
  layers: 5
  hidden size: 7168
  vocab size: 163840
  target_layer_ids: [0,1,3,4,5]
```

训练设计要求从完整 K3 target 捕获：

```text
target layers [7,23,51,67,83]
  → concatenate 5 × 7168
  → fc 35840 → 7168
  → initialize DSpark 5-layer draft KV
```

当前 smoke checkpoint 仅修改了 `target_layer_ids` 以适配 6 层 target；
这只解决索引和 shape，不能把已训练 FC/Draft/Markov 权重变成适配浅层
hidden 分布的新权重。真实 gamma=7 验证中 draft token 数量和内容都正常，
但所有 proposal 都未命中 target：

```text
64 completion tokens
63 verify rounds
441 proposed drafts = 63 × 7
0 correct drafts
accepted length = 64 / 63 = 1.015873...
```

因此 accepted length≈1 的根因是模型权重不匹配，不是 prefix cache、
NPU graph、acceptance 统计或 draft proposal 没有执行。要验证质量必须：

1. 使用与完整 K3 target、原始层 tap 完全配套的 DSpark checkpoint；或
2. 针对 6 层 target 和 `[0,1,3,4,5]` hidden 分布重新训练/蒸馏 drafter。

仍未覆盖：

- `cap-accept` 和经过真实 SPS table 驱动的 `compact`。
- 真实 Confidence head、SPS/STS 校准。
- 完整匹配权重下的 acceptance 与吞吐收益。
- Torch fallback 下沉后的 NPU kernel 性能。

---

## 20. 推荐验证顺序

1. 找到完整且匹配的 K3 target + DSpark checkpoint，恢复
   `[7,23,51,67,83]` layer taps。
2. 与关闭 speculative 的同一 K3 target 做 greedy 输出一致性验证。
3. 记录真实 accept histogram、吞吐和 graph replay 比例。
4. 验证 `cap-accept` 和带真实 SPS table 的 `compact`。
5. 加载真实 confidence head 和 STS calibration。
6. 将四个 Torch fallback 按性能优先级下沉到 `sgl-kernel-npu`。

---

## 21. 其他说明

启动脚本中的注释已从通用 MTP 含义明确为 DSpark verify 配置；建议保持：

```bash
# dspark
export SGLANG_RAGGED_VERIFY_MODE=static
```

`SGLANG_RAGGED_VERIFY_MODE` 当前控制 DSpark Verify Planner，不是一个独立的
MTP 算法开关。
