from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Set, Tuple

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.deepseek_v4 import fused_rope, rmsnorm_self
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_cp_split,
    get_nsa_prefill_cp_rank,
    get_nsa_prefill_cp_size,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_round_robin_split,
    nsa_cp_round_robin_split_q_seqs_cpu,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    attn_tp_all_gather,
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
    attn_tp_all_reduce,
    dp_gather_replicate,
    get_attention_tp_group,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import (
    compile_in_capture_mode,
    get_is_capture_mode,
)
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    get_bool_env_var,
    get_current_device_stream_fast,
    log_info_on_rank0,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

# torch.ops.custom.npu_hc_pre / npu_hc_post / inplace_partial_rotary_mul are
# registered as a side-effect of importing the `custom_ops` wheel that ships
# with the Ascend cann-8.5.0-a3 image. Import once at module load (NPU only)
# so the per-callsite re-imports inside hc_pre / hc_post can drop.
if _is_npu:
    import custom_ops  # noqa: F401  registers torch.ops.custom.*

logger = logging.getLogger(__name__)

_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
_DSV4_NPU_CP_VALUE_DEBUG_CONFIG_LOGGED = False


def _dsv4_npu_layer_filter_matches(
    layer_filter: Optional[str],
    layer_id: int,
    num_hidden_layers: Optional[int] = None,
) -> bool:
    if not layer_filter:
        return False
    for item in layer_filter.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "last":
            if num_hidden_layers is not None and layer_id == num_hidden_layers - 1:
                return True
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            if start.strip().isdigit() and end.strip().isdigit():
                if int(start) <= layer_id <= int(end):
                    return True
            continue
        if item.isdigit() and layer_id == int(item):
            return True
    return False


def _nsa_cp_round_robin_local_token_count(forward_batch: "ForwardBatch"):
    extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_seq_lens_cpu is None:
        return None
    local_q_lens_cpu, _ = nsa_cp_round_robin_split_q_seqs_cpu(
        extend_seq_lens_cpu,
        keep_zeros=True,
    )
    return int(sum(local_q_lens_cpu))


def _dsv4_npu_cp_value_debug_enabled(forward_batch: "ForwardBatch") -> bool:
    enabled = (
        get_bool_env_var("SGLANG_DSV4_NPU_CP_VALUE_DEBUG", "False")
        or get_bool_env_var("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_ALL_LAYERS", "False")
        or bool(os.environ.get("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_LAYERS"))
    )
    if not enabled:
        return False
    if nsa_use_prefill_cp(forward_batch):
        return True
    return (
        forward_batch.forward_mode.is_decode()
        and get_bool_env_var("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_DECODE", "False")
    )


def _dsv4_npu_log_tensor_stats(
    name: str,
    tensor: Optional[torch.Tensor],
    forward_batch: "ForwardBatch",
    *,
    layer_id: Optional[int] = None,
) -> None:
    if not _dsv4_npu_cp_value_debug_enabled(forward_batch):
        return
    global _DSV4_NPU_CP_VALUE_DEBUG_CONFIG_LOGGED
    if not _DSV4_NPU_CP_VALUE_DEBUG_CONFIG_LOGGED:
        logger.warning(
            "DSV4 NPU CP value debug active: layers=%s all_layers=%s mode=%s",
            os.environ.get("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_LAYERS"),
            get_bool_env_var("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_ALL_LAYERS", "False"),
            forward_batch.forward_mode,
        )
        _DSV4_NPU_CP_VALUE_DEBUG_CONFIG_LOGGED = True
    if tensor is None:
        logger.warning("DSV4 NPU CP value debug: %s is None", name)
        return
    try:
        data = tensor.detach()
        flat = data.float().reshape(-1)
        if flat.numel() == 0:
            logger.warning(
                "DSV4 NPU CP value debug: %s layer=%s shape=%s dtype=%s empty",
                name,
                layer_id,
                tuple(data.shape),
                data.dtype,
            )
            return
        finite = torch.isfinite(flat)
        finite_count = int(finite.sum().item())
        if finite_count > 0:
            finite_flat = flat[finite]
            abs_mean = float(finite_flat.abs().mean().item())
            abs_max = float(finite_flat.abs().max().item())
            mean = float(finite_flat.mean().item())
        else:
            abs_mean = abs_max = mean = float("nan")
        row_first_norm = row_last_norm = None
        if data.ndim > 0 and data.shape[0] > 0:
            rows = data.float().reshape(data.shape[0], -1)
            row_first_norm = float(rows[0].norm().item())
            row_last_norm = float(rows[-1].norm().item())
        sample = flat[: min(6, flat.numel())].cpu().tolist()
        logger.warning(
            "DSV4 NPU CP value debug: %s layer=%s mode=%s shape=%s dtype=%s "
            "finite=%s/%s mean=%.6e abs_mean=%.6e abs_max=%.6e "
            "row_norm(first,last)=%s,%s sample=%s",
            name,
            layer_id,
            forward_batch.forward_mode,
            tuple(data.shape),
            data.dtype,
            finite_count,
            flat.numel(),
            mean,
            abs_mean,
            abs_max,
            row_first_norm,
            row_last_norm,
            sample,
        )
    except Exception as exc:
        logger.warning("DSV4 NPU CP value debug failed for %s: %s", name, exc)


def _dsv4_npu_log_logits_stats(
    name: str,
    logits: Optional[torch.Tensor],
    forward_batch: "ForwardBatch",
) -> None:
    if not _dsv4_npu_cp_value_debug_enabled(forward_batch) or logits is None:
        return
    _dsv4_npu_log_tensor_stats(name, logits, forward_batch)
    try:
        if logits.numel() == 0 or logits.ndim < 2:
            return
        row = logits[0].detach().float()
        top_vals, top_ids = torch.topk(row, k=min(8, row.shape[-1]))
        logger.warning(
            "DSV4 NPU CP value debug: %s top ids=%s vals=%s",
            name,
            top_ids.cpu().tolist(),
            top_vals.cpu().tolist(),
        )
    except Exception as exc:
        logger.warning("DSV4 NPU CP logits topk debug failed for %s: %s", name, exc)


def _dsv4_npu_log_tensor_stats_no_batch(
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    layer_id: Optional[int] = None,
) -> None:
    if tensor is None:
        logger.warning("DSV4 NPU debug: %s layer=%s is None", name, layer_id)
        return
    try:
        data = tensor.detach()
        flat = data.float().reshape(-1)
        if flat.numel() == 0:
            logger.warning(
                "DSV4 NPU debug: %s layer=%s shape=%s dtype=%s empty",
                name,
                layer_id,
                tuple(data.shape),
                data.dtype,
            )
            return
        finite = torch.isfinite(flat)
        finite_count = int(finite.sum().item())
        if finite_count > 0:
            finite_flat = flat[finite]
            abs_mean = float(finite_flat.abs().mean().item())
            abs_max = float(finite_flat.abs().max().item())
            mean = float(finite_flat.mean().item())
        else:
            abs_mean = abs_max = mean = float("nan")
        row_first_norm = row_last_norm = None
        if data.ndim > 0 and data.shape[0] > 0:
            rows = data.float().reshape(data.shape[0], -1)
            row_first_norm = float(rows[0].norm().item())
            row_last_norm = float(rows[-1].norm().item())
        sample = flat[: min(6, flat.numel())].cpu().tolist()
        logger.warning(
            "DSV4 NPU debug: %s layer=%s shape=%s dtype=%s finite=%s/%s "
            "mean=%.6e abs_mean=%.6e abs_max=%.6e row_norm(first,last)=%s,%s "
            "sample=%s",
            name,
            layer_id,
            tuple(data.shape),
            data.dtype,
            finite_count,
            flat.numel(),
            mean,
            abs_mean,
            abs_max,
            row_first_norm,
            row_last_norm,
            sample,
        )
    except Exception as exc:
        logger.warning("DSV4 NPU debug failed for %s layer=%s: %s", name, layer_id, exc)


def _dsv4_npu_layer_detail_debug_enabled(
    forward_batch: "ForwardBatch",
    layer_id: int,
    num_hidden_layers: int,
) -> bool:
    if not _dsv4_npu_cp_value_debug_enabled(forward_batch):
        return False
    layer_filter = os.environ.get("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_LAYERS")
    if layer_filter:
        return _dsv4_npu_layer_filter_matches(
            layer_filter, layer_id, num_hidden_layers
        )
    if get_bool_env_var("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_ALL_LAYERS", "False"):
        return True
    return layer_id == 0 or layer_id == num_hidden_layers - 1


def _v4_rope_inplace_npu(
    q_rope: torch.Tensor,
    kv_rope: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """In-place interleaved RoPE for V4 — torch fallback used on NPU.

    Mirrors main's CUDA `fused_rope` kernel: consecutive (even, odd) pairs
    of x form complex pairs, with `freqs_cis` a complex tensor where
    `freqs_cis.real[t, k]` = cos(theta_{t,k}), `freqs_cis.imag` = sin(...)
    indexed by frequency pair k in [0, rope_dim/2).

    NOTE on V4-Flash YARN `mscale`: when the model was trained with the
    YARN magnitude-scale `mscale` ≠ 1.0, the cos/sin values stored in
    `freqs_cis` MUST already be pre-multiplied by `mscale` at precompute
    time — see `deepseek_v4_rope.precompute_freqs_cis`. This function
    just reads what's stored; it does NOT apply mscale here.

    Prefer the NPU-native `torch.ops.custom.inplace_partial_rotary_mul`
    (same kernel iforgetmyname/dsv4_release uses) — torch fallback differs
    by ~1 ULP per element vs the kernel because torch does bf16*bf16 muls
    with bf16 accumulation while the NPU kernel accumulates in fp32.
    43 layers × (Q + K) = 86 rope calls compound the drift; with the torch
    fallback some prompts (those with marginal argmax decisions) flip
    tokens vs iforgetmyname.
    """
    if (
        _is_npu
        and hasattr(torch.ops, "custom")
        and hasattr(torch.ops.custom, "inplace_partial_rotary_mul")
    ):
        # Build cos/sin caches in the layout the kernel expects:
        # (T, 1, 1, rope_dim) with each freq pair value repeated twice for
        # the interleaved pairing convention.
        cos_half = freqs_cis.real[positions]  # (T, rope_dim/2)
        sin_half = freqs_cis.imag[positions]
        if inverse:
            sin_half = -sin_half
        cos_full = cos_half.repeat_interleave(2, dim=-1).to(q_rope.dtype)
        sin_full = sin_half.repeat_interleave(2, dim=-1).to(q_rope.dtype)
        rope_dim = cos_full.shape[-1]
        # repeat_interleave produces a contiguous tensor, so the .view()
        # below already returns a contiguous result — no .contiguous() needed.
        cos4 = cos_full.view(-1, 1, 1, rope_dim)
        sin4 = sin_full.view(-1, 1, 1, rope_dim)
        # q_rope: (T, n_heads, rope_dim) → (T, 1, n_heads, rope_dim) view
        # kv_rope: (T, 1, rope_dim) → (T, 1, 1, rope_dim) view
        q_view = q_rope.unsqueeze(1)
        torch.ops.custom.inplace_partial_rotary_mul(
            q_view,
            cos4,
            sin4,
            rotary_mode="interleave",
            partial_slice=[0, rope_dim],
        )
        if kv_rope is not None:
            if kv_rope.dim() == 3:
                kv_view = kv_rope.unsqueeze(1)
            else:
                kv_view = kv_rope.view(-1, 1, 1, rope_dim)
            torch.ops.custom.inplace_partial_rotary_mul(
                kv_view,
                cos4,
                sin4,
                rotary_mode="interleave",
                partial_slice=[0, rope_dim],
            )
        return

    # Torch fallback (CUDA tests, or NPU images without the custom op).
    cos_full = freqs_cis.real
    sin_full = freqs_cis.imag
    cos = cos_full[positions].to(q_rope.dtype)
    sin = sin_full[positions].to(q_rope.dtype)

    def _apply(x: torch.Tensor) -> None:
        pair = x.view(*x.shape[:-1], -1, 2)
        re = pair[..., 0].clone()
        im = pair[..., 1].clone()
        c = cos
        s = sin
        while c.ndim < pair.ndim - 1:
            c = c.unsqueeze(-2)
            s = s.unsqueeze(-2)
        if inverse:
            pair[..., 0] = re * c + im * s
            pair[..., 1] = im * c - re * s
        else:
            pair[..., 0] = re * c - im * s
            pair[..., 1] = re * s + im * c

    _apply(q_rope)
    if kv_rope is not None:
        _apply(kv_rope)


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight

    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)

    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class MQALayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_nsa_prefill_cp_size()
            assert attn_tp_size == 1, (
                "DeepSeek V4 context-parallel token splitting requires "
                "attn_tp_size == 1 (each CP rank is its own TP singleton). "
                f"Got attn_tp_size={attn_tp_size}. "
                "Check that tp_size, dp_size, and attn_cp_size are configured "
                "so that attn_tp_size = tp_size // dp_size // attn_cp_size == 1."
            )
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else config.compress_ratios[layer_id]
        )
        # V4-Flash uses ratio=1 for the dense (uncompressed) edge layers
        # (e.g. [1, 1, 4, 128, ..., 1] for the 43-layer Flash variant), in
        # addition to the {0, 4, 128} the original V4 model ships. Treat 0
        # and 1 the same throughout: neither has a Compressor/C4Indexer and
        # neither uses the compress_rope_theta / yarn original_seq_len.
        assert compress_ratio in (
            0,
            1,
            4,
            128,
        ), f"V4 compress_ratio: expected one of (0, 1, 4, 128), got {compress_ratio}"
        self.compress_ratio: Literal[0, 1, 4, 128] = compress_ratio

        assert self.head_dim == config.head_dim
        assert config.num_key_value_heads == 1

        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        rope_base = (
            config.compress_rope_theta
            if self.compress_ratio in (4, 128)
            else rope_theta
        )

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        # YARN correction applies to ALL layers regardless of compress_ratio.
        # iforgetmyname/dsv4_release's get_rope_wrapper always passes
        # `rope_scaling` to the rotary_emb constructor, so dense (ratio 0/1)
        # layers get the same YARN-corrected inv_freq as compressed layers
        # — only the rope base differs (rope_theta vs compress_rope_theta).
        # Earlier this branched on compress_ratio and zeroed the YARN
        # correction for dense layers; that left the 3 dense layers of
        # V4-Flash (positions 0, 1, 43 in the 44-layer model) with raw
        # extrapolation freqs while the model was trained with YARN —
        # causing token-1+ divergence even after the mscale magnitude fix.
        original_seq_len = rope_scaling["original_max_position_embeddings"]

        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
            # V4-Flash YARN magnitude-scale: V4-Flash config.json doesn't
            # include mscale / mscale_all_dim fields, so iforgetmyname's
            # DeepseekScalingRotaryEmbedding falls back to defaults
            # (mscale=1, mscale_all_dim=0) which yield self.mscale ≈ 1.2773
            # for factor=16. main's precompute_freqs_cis previously omitted
            # this scale entirely, producing rope magnitudes ~22% smaller
            # than what the model was trained for → semantically broken
            # output (231 step 5+ test: tokens like `[19,16,19,16,343,...]`
            # = "1.1. (C.dir," vs the correct `[54994,1060,...]` =
            # "jumps over the lazy dog" for prompt "The quick brown fox").
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio in (4, 128):
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                )

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        if self.fuse_wqa_wkv:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size == get_tensor_model_parallel_world_size() and attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.overlap_store_cache = envs.SGLANG_OPT_USE_OVERLAP_STORE_CACHE.get()
        self.use_jit_norm = envs.SGLANG_OPT_USE_JIT_NORM.get()

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q_lora = q
        return q_lora

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if _is_npu:
            # iforgetmyname/dsv4_release uses torch_npu.npu_rms_norm with an
            # all-ones dummy weight (q_b_norm_dummy_weight) for this second
            # RMS post wq_b. Our triton rms_normalize_triton produces ULP-
            # different bf16 results vs the NPU-native kernel; with 43
            # layers of cascade, that 1-2% per-layer drift flips argmax in
            # tokens 5+. Match exact iforgetmyname behavior here.
            import torch_npu

            _dummy = q.new_ones(q.shape[-1])
            q = torch_npu.npu_rms_norm(q, _dummy, self.eps)[0]
        elif self.use_jit_norm:
            q = rmsnorm_self(q, self.eps)
        else:
            q = rms_normalize_triton(q, self.eps)
        if positions is not None:
            fused_rope(
                q[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(q[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return q

    def _compute_kv(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        kv = self.kv_norm(kv)
        if positions is not None:
            fused_rope(
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(kv[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4AttnBackend,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            qkv_a_ready = current_stream.record_event()

        q_lora = self._compute_q_a(x, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            kv = self._compute_kv(x, positions, qkv_a=qkv_a)
            if self.overlap_store_cache:
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )

        del qkv_a

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_lora, positions)
        if q_out is not None:
            q_out.copy_(q)

        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q, kv

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4AttnBackend,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            q = qkv_a[..., : self.q_lora_rank]
            kv = qkv_a[..., self.q_lora_rank :]
            del qkv_a
        else:
            kv, _ = self.wkv(x)
            q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q_lora = q
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if _is_npu:
            # iforgetmyname/dsv4_release uses torch_npu.npu_rms_norm with an
            # all-ones dummy weight (q_b_norm_dummy_weight) for this second
            # RMS post wq_b. Our triton rms_normalize_triton produces ULP-
            # different bf16 results vs the NPU-native kernel; with 43
            # layers of cascade, that 1-2% per-layer drift flips argmax in
            # tokens 5+. Match exact iforgetmyname behavior here.
            import torch_npu

            _dummy = q.new_ones(q.shape[-1])
            q = torch_npu.npu_rms_norm(q, _dummy, self.eps)[0]
        elif self.use_jit_norm:
            q = rmsnorm_self(q, self.eps)
        else:
            q = rms_normalize_triton(q, self.eps)

        kv = self.kv_norm(kv)

        if _is_npu:
            # `fused_rope` is a tvm_ffi-compiled CUDA kernel; NPU images don't
            # ship tvm_ffi, so do the same operation (in-place rotary on the
            # last qk_rope_head_dim of q and kv) with a torch fallback. The
            # interleaved pair convention here matches fused_rope's CUDA
            # source (consecutive even/odd indices form a complex pair).
            _v4_rope_inplace_npu(
                q[..., -self.qk_rope_head_dim :],
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                self.freqs_cis,
                positions,
            )
        else:
            fused_rope(
                q[..., -self.qk_rope_head_dim :],
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                self.freqs_cis,
                positions=positions,
            )

        if self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch):
            kv = cp_all_gather_rerange_output(
                kv.contiguous(),
                self.cp_size,
                forward_batch,
                get_current_device_stream_fast(),
            )
            _dsv4_npu_log_tensor_stats(
                "mqa.gathered_kv",
                kv,
                forward_batch,
                layer_id=self.layer_id,
            )

        if self.overlap_store_cache:
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )

        if self.indexer is not None:
            self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch)
        if self.compressor is not None:
            compressor_x = x
            if (
                self.nsa_enable_prefill_cp
                and nsa_use_prefill_cp(forward_batch)
                and is_nsa_prefill_cp_round_robin_split()
            ):
                # Round-robin CP: each rank holds a strided token shard, but the
                # compressor needs contiguous full-sequence activations to produce
                # correct compressed KV. All-gather and rerange before compressing.
                # In-seq-split CP does not need this because each rank's shard is
                # already a contiguous subsequence.
                compressor_x = cp_all_gather_rerange_output(
                    x.contiguous(),
                    self.cp_size,
                    forward_batch,
                    get_current_device_stream_fast(),
                )
            attn_backend.forward_core_compressor(
                compressor_x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )

        if q_out is not None:
            q_out.copy_(q)
        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            return x

        attn_backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4AttnBackend)

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch))
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            q, kv = self._forward_prepare_multi_stream(
                x, positions, forward_batch, attn_backend, q_out
            )
        else:
            q, kv = self._forward_prepare(
                x, positions, forward_batch, attn_backend, q_out
            )

        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=kv,
            v=kv,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=not self.overlap_store_cache,
        )
        _dsv4_npu_log_tensor_stats(
            "mqa.attn_out",
            o,
            forward_batch,
            layer_id=self.layer_id,
        )
        o = o[:, tp_slice, :]
        if _is_npu:
            _v4_rope_inplace_npu(
                o[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions,
                inverse=True,
            )
        else:
            fused_rope(
                o[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
                inverse=True,
            )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D).contiguous(),
                group_size=128,
            )
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))
        if self.tp_size > 1 and self.tp_size < get_tensor_model_parallel_world_size():
            o = attn_tp_all_reduce(o)

        return o


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.num_hidden_layers = config.num_hidden_layers
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=alt_streams[0] if alt_streams is not None else None,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
    ):
        """If *norm* is given and the TileLang path is active, the returned
        hidden_states are already post-norm (the norm is fused into the kernel)."""

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb, False

        # NPU fast path: the cann8.5.0-a3 image ships a `custom_ops` wheel
        # that registers torch.ops.custom.npu_hc_pre — a fused RMS-norm +
        # linear projection + sinkhorn iteration kernel that returns the
        # same (post, comb, y) triple the rest of this method computes via
        # tilelang/deep_gemm/torch. Use it instead of mhc.py on Ascend (where
        # tilelang isn't available and the torch fallback would be slow).
        if _is_npu:
            # Note the return order: (y, post, comb) — y is the (T, hidden)
            # mixed activation, post / comb are the hc_post inputs. The
            # fused kernel emits y in fp32 (sinkhorn iterates in fp32), so
            # cast back to the input dtype before the downstream
            # aclnnRmsNorm (which has no x=fp32 / gamma=bf16 overload).
            y, post, comb = torch.ops.custom.npu_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult=self.hc_mult,
                hc_sinkhorn_iters=self.hc_sinkhorn_iters,
                norm_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
            )
            # npu_hc_pre uses norm_eps for sinkhorn's internal RMS only; it does
            # not fold input_layernorm. Return norm_fused=False so the caller
            # applies the layernorm itself, matching the deepgemm/torch paths.
            return y.to(dtype), post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            return y, post.squeeze(-1), comb, norm is not None

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            import deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        stage: str = "unknown",
    ):

        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        # NPU fast path mirroring hc_pre: torch.ops.custom.npu_hc_post is
        # the fused output replication + mixing kernel shipped in the
        # cann8.5.0-a3 image's custom_ops wheel.
        if _is_npu:
            torch_layers = os.environ.get("SGLANG_DSV4_NPU_TORCH_HC_POST_LAYERS")
            debug_layers = os.environ.get("SGLANG_DSV4_NPU_HC_POST_DEBUG_LAYERS")
            use_torch_hc_post = _dsv4_npu_layer_filter_matches(
                torch_layers,
                self.layer_id,
                self.num_hidden_layers,
            )
            debug_hc_post = _dsv4_npu_layer_filter_matches(
                debug_layers,
                self.layer_id,
                self.num_hidden_layers,
            ) or use_torch_hc_post

            npu_out = None
            if not use_torch_hc_post or debug_hc_post:
                npu_out = torch.ops.custom.npu_hc_post(x, residual, post, comb)
            if use_torch_hc_post or debug_hc_post:
                torch_out = hc_post_torch_impl(x, residual, post, comb)
                if debug_hc_post:
                    log_prefix = f"hc_post.{stage}"
                    _dsv4_npu_log_tensor_stats_no_batch(
                        f"{log_prefix}.x", x, layer_id=self.layer_id
                    )
                    _dsv4_npu_log_tensor_stats_no_batch(
                        f"{log_prefix}.residual", residual, layer_id=self.layer_id
                    )
                    _dsv4_npu_log_tensor_stats_no_batch(
                        f"{log_prefix}.post", post, layer_id=self.layer_id
                    )
                    _dsv4_npu_log_tensor_stats_no_batch(
                        f"{log_prefix}.comb", comb, layer_id=self.layer_id
                    )
                    _dsv4_npu_log_tensor_stats_no_batch(
                        f"{log_prefix}.torch_out", torch_out, layer_id=self.layer_id
                    )
                    if npu_out is not None:
                        _dsv4_npu_log_tensor_stats_no_batch(
                            f"{log_prefix}.npu_out", npu_out, layer_id=self.layer_id
                        )
                        diff = (npu_out.float() - torch_out.float()).abs()
                        _dsv4_npu_log_tensor_stats_no_batch(
                            f"{log_prefix}.npu_minus_torch_abs",
                            diff,
                            layer_id=self.layer_id,
                        )
                if use_torch_hc_post:
                    return torch_out
            assert npu_out is not None
            return npu_out

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        return hc_post_torch_impl(x, residual, post, comb)

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        layer_debug = _dsv4_npu_layer_detail_debug_enabled(
            forward_batch,
            self.layer_id,
            self.num_hidden_layers,
        )
        clone_residual = _dsv4_npu_layer_filter_matches(
            os.environ.get("SGLANG_DSV4_NPU_CLONE_RESIDUAL_LAYERS"),
            self.layer_id,
            self.num_hidden_layers,
        )
        residual = hidden_states
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.input",
                hidden_states,
                forward_batch,
                layer_id=self.layer_id,
            )
        if clone_residual:
            residual = residual.clone()
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            norm=self.input_layernorm,
        )
        if not norm_fused:
            hidden_states = self.input_layernorm(hidden_states)
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.attn_normed",
                hidden_states,
                forward_batch,
                layer_id=self.layer_id,
            )
            _dsv4_npu_log_tensor_stats(
                "layer.attn_residual_after_hc_pre",
                residual,
                forward_batch,
                layer_id=self.layer_id,
            )

        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.attn_out",
                hidden_states,
                forward_batch,
                layer_id=self.layer_id,
            )

        hidden_states = self.hc_post(
            hidden_states, residual, post, comb, stage="attn"
        )
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.after_attn_hc_post",
                hidden_states,
                forward_batch,
                layer_id=self.layer_id,
            )
        residual = hidden_states
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.mlp_residual_input",
                residual,
                forward_batch,
                layer_id=self.layer_id,
            )
        if clone_residual:
            residual = residual.clone()
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm=self.post_attention_layernorm,
        )
        if not norm_fused:
            hidden_states = self.post_attention_layernorm(hidden_states)
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.mlp_normed",
                hidden_states,
                forward_batch,
                layer_id=self.layer_id,
            )
            _dsv4_npu_log_tensor_stats(
                "layer.mlp_residual_after_hc_pre",
                residual,
                forward_batch,
                layer_id=self.layer_id,
            )

        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()
            and get_attention_tp_size() > 1
            and not get_moe_a2a_backend().is_none()
        )
        if _use_cp:
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            cp_rank = get_nsa_prefill_cp_rank()
            cp_size = get_nsa_prefill_cp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            if is_nsa_prefill_cp_round_robin_split():
                local_num_tokens = hidden_states.shape[0]
                if input_ids.shape[0] > local_num_tokens:
                    input_ids = input_ids[:local_num_tokens].contiguous()
                elif input_ids.shape[0] < local_num_tokens:
                    raise RuntimeError(
                        "DeepSeek V4 CP input_ids split is shorter than local "
                        f"hidden states: input_ids={input_ids.shape[0]}, "
                        f"hidden_states={local_num_tokens}, layer_id={self.layer_id}."
                    )
            input_ids_global = input_ids
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            # hidden_states here follow TP_ATTN_FULL semantics: they are replicated
            # within an attention-TP group. Use replicate gather to avoid summing the
            # same activations across attention-TP ranks before entering MLP/MoE.
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_attention_tp_size(), get_attention_tp_rank()
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            input_ids = input_ids.tensor_split(s)[r].contiguous()
            input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.mlp_out",
                hidden_states,
                forward_batch,
                layer_id=self.layer_id,
            )
        if _use_tp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_attention_tp_group()),
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        hidden_states = self.hc_post(
            hidden_states, residual, post, comb, stage="mlp"
        )
        if layer_debug:
            _dsv4_npu_log_tensor_stats(
                "layer.final",
                hidden_states,
                forward_batch,
                layer_id=self.layer_id,
            )

        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(5)] if (_is_cuda or _is_hip) else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_nsa_prefill_cp_size()

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        if x.numel() > 0:
            from sglang.srt.layers.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        _dsv4_npu_log_tensor_stats("model.embed", hidden_states, forward_batch)

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            # Token ids are replicated within an attention-TP group. Use replicate
            # gather here to avoid summing duplicated ids when attention_tp_size > 1.
            dp_gather_replicate(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
            if is_nsa_prefill_cp_round_robin_split():
                local_num_tokens = _nsa_cp_round_robin_local_token_count(forward_batch)
                if local_num_tokens is not None:
                    if hidden_states.shape[0] > local_num_tokens:
                        if get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False"):
                            logger.warning(
                                "DSV4 NPU CP trims local padded shard before "
                                "layers: hidden_rows=%s, position_rows=%s, "
                                "local_real_rows=%s, extend_lens=%s",
                                hidden_states.shape[0],
                                positions.shape[0],
                                local_num_tokens,
                                getattr(forward_batch, "extend_seq_lens_cpu", None),
                            )
                        hidden_states = hidden_states[:local_num_tokens].contiguous()
                        positions = positions[:local_num_tokens].contiguous()
                    elif hidden_states.shape[0] < local_num_tokens:
                        raise RuntimeError(
                            "DeepSeek V4 CP local shard is shorter than expected: "
                            f"hidden_rows={hidden_states.shape[0]}, "
                            f"local_real_rows={local_num_tokens}, "
                            f"extend_lens={getattr(forward_batch, 'extend_seq_lens_cpu', None)}."
                        )
            _dsv4_npu_log_tensor_stats(
                "model.after_cp_split.hidden",
                hidden_states,
                forward_batch,
            )
            _dsv4_npu_log_tensor_stats(
                "model.after_cp_split.positions",
                positions,
                forward_batch,
            )

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
            )
            if _dsv4_npu_layer_detail_debug_enabled(
                forward_batch, i, len(self.layers)
            ):
                _dsv4_npu_log_tensor_stats(
                    "model.after_layer",
                    hidden_states,
                    forward_batch,
                    layer_id=i,
                )

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                get_current_device_stream_fast(),
            )
            _dsv4_npu_log_tensor_stats(
                "model.after_cp_gather",
                hidden_states,
                forward_batch,
            )

        pre_hc_head = hidden_states.flatten(1)
        _dsv4_npu_log_tensor_stats("model.pre_hc_head", pre_hc_head, forward_batch)

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)
        _dsv4_npu_log_tensor_stats("model.final_norm", hidden_states, forward_batch)

        return hidden_states, pre_hc_head


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_nsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE)
            }
        )

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_nsa_prefill_cp_rank()
            self.cp_size = get_nsa_prefill_cp_size()

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        get_global_server_args().disable_shared_experts_fusion = True
        log_info_on_rank0(
            logger,
            "DeepSeek V4 requires different clamping for shared and routed experts. "
            "Shared experts fusion optimization is disabled.",
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.nsa_enable_prefill_cp:
            assert (not _is_npu) or is_nsa_prefill_cp_round_robin_split(), (
                "DeepSeek V4 NPU CP currently supports only "
                "nsa_prefill_cp_mode=round-robin-split."
            )
            # Round-robin CP activates via forward_mode.is_prefill(); it does NOT
            # use can_nsa_cp_split (which checks is_context_parallel_extend() —
            # always False for round-robin) or prepare_context_parallel_metadata.
            # In-seq-split CP uses can_nsa_cp_split + prepare_context_parallel_metadata.
            if is_nsa_prefill_cp_round_robin_split():
                # Round-robin: no attn_cp_metadata needed; CP is driven by the
                # nsa_use_prefill_cp / can_nsa_prefill_cp_round_robin_split gates.
                pass
            elif can_nsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                # In-seq-split: build zigzag context-parallel metadata for all
                # devices. prepare_context_parallel_metadata sets attn_cp_metadata
                # which the flash-attention backend uses to partition tokens.
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds
            )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states

        logits_output = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=pre_hc_head,
        )
        _dsv4_npu_log_logits_stats(
            "logits.next_token",
            logits_output.next_token_logits,
            forward_batch,
        )
        return logits_output

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        layers = self.model.layers
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        if is_nextn:
            return
        for layer in self.model.layers:
            self_attn = layer.self_attn
            if (
                self_attn.compress_ratio in (4, 128)
                and not self_attn.compressor.ape_converted
            ):
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str,
        is_nextn: bool = False,
        num_hidden_layers: Optional[int] = None,
        scale_suffix: str = "weight_scale_inv",
    ) -> str:
        # ``scale_suffix`` controls how DeepSeek modelslim's ``.scale`` channel
        # gets renamed to match the in-model parameter:
        #  * "weight_scale_inv" — FP8 ckpts (deep_gemm convention, the default)
        #  * "weight_scale"     — INT8 / W8A8 ckpts saved by compressed-tensors
        # V4-Flash-W8A8 uses int-quantized + per-channel scale; without this
        # parameterisation the loader looks up gate_up_proj.weight_scale_inv
        # while the int8 quant scheme registers gate_up_proj.weight_scale.
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj." + scale_suffix
                    elif rest == "h_proj.scale":
                        rest = "h_proj." + scale_suffix
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        scale_target = "." + scale_suffix
        if "self_attn" in name:
            name = name.replace(".scale", scale_target)

        name = name.replace(".gate.tid2eid", ".topk.tid2eid")
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if "mlp" in name:
            name = name.replace(".scale", scale_target)

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        # Compressed-tensors saves int-quantized W8A8 ckpts with parameter
        # name `weight_scale` while FP8 ckpts save `weight_scale_inv`. The
        # remap below has to match whichever the model created — sniff it
        # off any existing scale parameter, falling back to the FP8 default.
        scale_suffix = "weight_scale_inv"
        for _name in params_dict:
            if _name.endswith(".weight_scale"):
                scale_suffix = "weight_scale"
                break
            if _name.endswith(".weight_scale_inv"):
                break  # default already correct

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        if not envs.SGLANG_OPT_FP8_WO_A_GEMM.get():
            weights = list(weights)
            exists_wo_a_scale = any(n.endswith(".wo_a.scale") for n, t in weights)
            if exists_wo_a_scale:
                logger.info("Execute dequant fp8 wo_a")
                weights = _dequant_fp8_wo_a(weights)
            else:
                logger.info("Skip dequant fp8 wo_a")

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                        scale_suffix=scale_suffix,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if ".norm." in name and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                bucket = cache_wqkv_a_weight.setdefault(param_name, {})
                                shard_key = "q" if is_q else "kv"
                                assert (
                                    shard_key not in bucket
                                ), f"duplicate shard {shard_key} for {param_name}"
                                bucket[shard_key] = loaded_weight
                                if len(bucket) == 2:
                                    fused_weight = torch.cat(
                                        [bucket["q"], bucket["kv"]], dim=0
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(param_name)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        # Hot weight reload (RL workflows). Use the device-agnostic module
        # accessor so this works on both CUDA/HIP and NPU.
        torch.get_device_module().empty_cache()
        torch.get_device_module().synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float32,
    ), f"expected fp8_e8m0fnu or float32, got {scale.dtype}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    return result.to(torch.bfloat16)


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()
