"""DeepSeek V4 attention backend on Ascend NPU.

This bridges sgl-project/sglang's V4 model code (which expects a backend
that mixes ``CompressorBackendMixin`` + ``C4IndexerBackendMixin`` on top of
``AttentionBackend``) with ``AscendAttnBackend`` (the NPU implementation
that knows nothing about V4's c4/c128 compress paths). The CUDA reference
is ``DeepseekV4AttnBackend``; this class is its NPU counterpart.

Strategy:

* Inherit from ``AscendAttnBackend`` plus the two V4 mixins. The mixins
  give us ``forward_compress`` / ``forward_core_compressor`` / ``forward_c4_indexer``
  signatures the model calls. Their default implementations call CUDA JIT
  kernels (``compress_forward``, ``compress_fused_norm_rope_inplace``,
  ``act_quant``, ``rotate_activation``, etc.); on NPU each of these has to
  be replaced with an ATB / torch_npu / pure-torch equivalent. We override
  one method at a time as we hit them at runtime.

* ``init_forward_metadata`` has to compute both the regular ascend metadata
  and the V4 ``DSV4Metadata`` with ``DSV4AttnMetadata`` + indexer metadata
  + c4/c128 compress metadata. We delegate the ascend half and add a thin
  V4 layer on top.

* ``forward()`` accepts V4-specific kwargs (``compress_ratio``, ``attn_sink``,
  ``save_kv_cache``). For ``compress_ratio==0`` (regular MQA layers) we
  delegate to ``AscendAttnBackend.forward``; for 4 / 128 we have to route
  to the c4 / c128 sparse path.

This file deliberately leaves the harder methods unimplemented behind
``NotImplementedError`` with explicit messages — the goal is to surface
exact method names + arguments at first NPU forward, then fill them in.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

# custom_ops registers torch.ops.custom.npu_sparse_attn_sharedkv_metadata,
# npu_sparse_attn_sharedkv, npu_quant_lightning_indexer and friends. The
# V4 ascend backend has no pure-torch fallback for those ops, so if the
# import fails we must fail fast with a clear message rather than crash
# later with an opaque AttributeError on torch.ops.custom.<name>.
try:
    import custom_ops  # noqa: F401
except ImportError as e:
    raise ImportError(
        "DeepSeek-V4 ascend attention backend requires the `custom_ops` "
        "wheel that ships with the Ascend cann-8.5.0-a3 image (registers "
        "torch.ops.custom.npu_sparse_attn_sharedkv_*, "
        "npu_quant_lightning_indexer, npu_hc_pre/post, etc.). The package "
        "is normally at /usr/local/python*/site-packages/custom_ops. "
        f"Original ImportError: {e}"
    ) from e

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.layers.attention.dsv4.compressor import CompressorBackendMixin
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_prefill_cp_round_robin_split,
    get_nsa_prefill_cp_total_len,
    nsa_cp_round_robin_split_data,
    nsa_cp_round_robin_split_q_seqs_cpu,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _dsv4_npu_layer_filter() -> Optional[str]:
    return (
        os.environ.get("SGLANG_DSV4_NPU_CP_VERIFY_LAYERS")
        or os.environ.get("SGLANG_DSV4_NPU_DEBUG_LAYERS")
        or os.environ.get("SGLANG_DSV4_NPU_CP_VALUE_DEBUG_LAYERS")
    )


def _dsv4_npu_should_log_layer(
    layer_id: int,
    num_hidden_layers: Optional[int] = None,
) -> bool:
    layer_filter = _dsv4_npu_layer_filter()
    if not layer_filter:
        return True
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


def _stub(method_name: str):
    raise NotImplementedError(
        f"DeepseekV4AscendAttnBackend.{method_name} is not implemented yet on NPU. "
        "The CUDA reference is in deepseek_v4_backend.py / dsv4/{compressor,indexer}.py; "
        "the NPU port has to either (a) call into torch_npu / ATB / sgl_kernel_npu "
        "for the corresponding fused op, or (b) provide a pure-torch fallback."
    )


def _dsv4_prepare_attn_sink(
    attn_sink: Optional[torch.Tensor],
    layer_id: int,
    num_hidden_layers: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if attn_sink is None:
        return None

    cp_verify = get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False")
    if get_bool_env_var("SGLANG_DSV4_NPU_ZERO_ATTN_SINK", "False"):
        if cp_verify and _dsv4_npu_should_log_layer(layer_id, num_hidden_layers):
            logger.warning(
                "DSV4 NPU uses zeroed attn_sink: layer=%s shape=%s dtype=%s",
                layer_id,
                tuple(attn_sink.shape),
                attn_sink.dtype,
            )
        return torch.zeros_like(attn_sink)

    if cp_verify and _dsv4_npu_should_log_layer(layer_id, num_hidden_layers):
        try:
            flat = attn_sink.detach().float().reshape(-1)
            finite = torch.isfinite(flat)
            finite_count = int(finite.sum().item())
            if finite_count > 0:
                finite_flat = flat[finite]
                min_value = float(finite_flat.min().item())
                max_value = float(finite_flat.max().item())
                abs_max = float(finite_flat.abs().max().item())
            else:
                min_value = max_value = abs_max = float("nan")
            sample = flat[: min(6, flat.numel())].cpu().tolist()
            logger.warning(
                "DSV4 NPU attn_sink: layer=%s shape=%s dtype=%s "
                "finite=%s/%s min=%.6e max=%.6e abs_max=%.6e sample=%s",
                layer_id,
                tuple(attn_sink.shape),
                attn_sink.dtype,
                finite_count,
                flat.numel(),
                min_value,
                max_value,
                abs_max,
                sample,
            )
        except Exception as exc:
            logger.warning(
                "DSV4 NPU attn_sink debug failed: layer=%s error=%s",
                layer_id,
                exc,
            )
    return attn_sink


def _build_hadamard_matrix(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Sylvester-construction Walsh-Hadamard matrix of size n × n.

    n must be a power of 2 (asserted by callers). Caches per (n, dtype, device)
    on the function so repeated calls within a forward batch don't rebuild.
    """
    cache = _build_hadamard_matrix._cache  # type: ignore[attr-defined]
    key = (n, dtype, str(device))
    if key in cache:
        return cache[key]
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.size(0) < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    H = H.to(dtype=dtype, device=device).contiguous()
    cache[key] = H
    return H


_build_hadamard_matrix._cache = {}  # type: ignore[attr-defined]


def _compute_c4_q_npu(
    c4_indexer,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """NPU equivalent of ``C4Indexer.compute_q``.

    ``compute_q`` does:
        q, _ = wq_b(q_lora)
        q = q.view(-1, n_local_heads, head_dim)
        fused_rope(q[..., -rope_head_dim:], None, freqs_cis, positions=...)
        q = rotate_activation(q)            # triton hadamard_transform

    On NPU, ``fused_rope`` is a tvm_ffi CUDA kernel and ``rotate_activation``
    is a triton hadamard. Replace with ``_v4_rope_inplace_npu`` and a torch
    Walsh-Hadamard matmul. Note: Sylvester ordering may not match the triton
    kernel's ordering — final consumer (``npu_quant_lightning_indexer``) is
    insensitive to the basis since both q and k are rotated by the same H.
    """
    from sglang.srt.models.deepseek_v4 import _v4_rope_inplace_npu

    q, _ = c4_indexer.wq_b(q_lora)
    q = q.view(-1, c4_indexer.n_local_heads, c4_indexer.head_dim)
    _v4_rope_inplace_npu(
        q[..., -c4_indexer.rope_head_dim :],
        None,
        c4_indexer.freqs_cis,
        positions,
    )
    H = _build_hadamard_matrix(c4_indexer.head_dim, torch.float32, q.device)
    scale = c4_indexer.head_dim**-0.5
    q_f32 = q.to(torch.float32)
    q_rotated = torch.matmul(q_f32, H) * scale
    return q_rotated.to(torch.bfloat16)


class DeepseekV4AscendAttnBackend(
    AscendAttnBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    """V4 attention dispatcher for Ascend NPU.

    Method resolution order is intentional: AscendAttnBackend ships the
    NPU-side ``init_forward_metadata`` / ``forward_extend`` / ``forward_decode``
    surface; the V4 mixins only add the c4/c128 compress + c4 indexer
    helpers. When both define a method (e.g. ``forward``), MRO picks
    Ascend's, which is what we want for the regular MQA path.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        speculative_step_id: int = 0,
    ):
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        # Pull the V4-specific config that compute_kernel_metadata needs.
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        cfg = model_runner.model_config
        self._dsv4_config = cfg
        tp_size = get_attention_tp_size()
        self._dsv4_q_head_num = cfg.num_attention_heads // tp_size
        self._dsv4_kv_head_num = 1  # V4 MQA / latent
        # V4-Flash config.json sets head_dim=512 directly (qk_nope_head_dim is
        # null in HF config); mirror iforgetmyname/dsv4_release which uses
        # self.config.head_dim verbatim for the metadata kernel arg.
        self._dsv4_head_dim = cfg.head_dim
        hf = getattr(cfg, "hf_config", cfg)
        self._dsv4_index_topk = getattr(hf, "index_topk", 512)
        self._dsv4_index_n_heads = getattr(hf, "index_n_heads", 64)
        self._dsv4_index_head_dim = getattr(hf, "index_head_dim", 128)
        self._dsv4_num_hidden_layers = getattr(hf, "num_hidden_layers", None)
        self._dsv4_compress_ratios = getattr(hf, "compress_ratios", None)
        self._dsv4_has_c4 = (
            self._dsv4_compress_ratios is not None and 4 in self._dsv4_compress_ratios
        )
        self._dsv4_has_c128 = (
            self._dsv4_compress_ratios is not None and 128 in self._dsv4_compress_ratios
        )
        self._dsv4_sliding_window_size = (
            cfg.sliding_window_size if cfg.sliding_window_size is not None else 128
        )
        self.dsv4_cp_prefill_row_kernel_metadata = {}
        self._dsv4_cp_verify_init_metadata_logged = False
        if get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False"):
            logger.warning(
                "DSV4 NPU CP verify is active: backend=%s, file=%s, "
                "tp_size=%s, page_size=%s, sliding_window=%s",
                type(self).__name__,
                __file__,
                tp_size,
                self.page_size,
                self._dsv4_sliding_window_size,
            )

    # ------------------------------------------------------------------
    # V4-specific metadata + dispatch — all stubbed pending real impls.
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        super().init_forward_metadata(forward_batch)
        fm = self.forward_metadata
        self.dsv4_cp_prefill_row_kernel_metadata = {}
        self.dsv4_cp_prefill_batch_kernel_metadata = {}
        cp_verify = get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False")
        if cp_verify and not self._dsv4_cp_verify_init_metadata_logged:
            from sglang.srt.layers.attention.nsa.utils import (
                get_nsa_prefill_cp_rank,
                get_nsa_prefill_cp_size,
                is_nsa_enable_prefill_cp,
                is_nsa_prefill_cp_round_robin_split,
            )

            logger.warning(
                "DSV4 NPU CP metadata probe: backend=%s, mode=%s, "
                "is_prefill=%s, nsa_cp=%s, rr=%s, can_rr_split=%s, "
                "cp_rank=%s, cp_size=%s, batch_size=%s, seq_lens=%s, "
                "extend_lens=%s",
                type(self).__name__,
                forward_batch.forward_mode,
                forward_batch.forward_mode.is_prefill(),
                is_nsa_enable_prefill_cp(),
                is_nsa_prefill_cp_round_robin_split(),
                can_nsa_prefill_cp_round_robin_split(forward_batch),
                get_nsa_prefill_cp_rank(),
                get_nsa_prefill_cp_size(),
                forward_batch.batch_size,
                getattr(forward_batch, "seq_lens_cpu", None),
                getattr(forward_batch, "extend_seq_lens_cpu", None),
            )
            self._dsv4_cp_verify_init_metadata_logged = True

        # DP-attention IDLE ranks get a padded batch (bs>0) but seq_lens are
        # all zero. The sparse-attn metadata kernel
        # (npu_sparse_attn_sharedkv_metadata) doesn't accept this shape; even
        # after clamping seqused_kv it tries to read the request's page table
        # at positions that were never written, which surfaces as an AICPU
        # exception (errcode 0x2a / runtime 507018) on the next sync.
        # The rest of the V4 backend already treats IDLE as a no-op (see
        # forward_compress / forward_c4_indexer below), so we mirror that
        # contract here: stash empty-but-typed defaults on fm so any later
        # attribute access stays well-defined, then return without invoking
        # any sparse-attn metadata kernels.
        if forward_batch.forward_mode.is_idle():
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None
            fm.kernel_metadata = {}
            return

        # Build TND cu_seqlens_q (= cumulative QUERY seq lens, int32 device tensor).
        # The kernel uses cu_seqlens_q to slice the q tensor by request, so
        # the per-request length here must equal the per-request token count
        # in q — NOT the KV/context length.
        #
        #   extend / prefill: q has extend_seq_lens_cpu tokens per request →
        #                     cumsum(extend_seq_lens_cpu).
        #   decode:           q has exactly 1 new token per request → [1, 1, ..., 1].
        #   target_verify /
        #   draft_extend:     q has speculative_num_draft_tokens per request.
        #
        # Earlier this branch fell back to `forward_batch.seq_lens_cpu` (the
        # full KV length) on the non-extend path, which made the kernel slice
        # q at offset = full_seq_len while q.shape[0] = batch_size for decode.
        # That is the V4-NPU root cause of token-1+ divergence — kernel
        # metadata says q has e.g. 257 tokens but q tensor only has 1.
        device = forward_batch.seq_lens.device
        if forward_batch.forward_mode.is_extend():
            seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            if isinstance(seq_lens_cpu, list):
                seq_lens_cpu = torch.tensor(seq_lens_cpu, dtype=torch.int32)
            else:
                seq_lens_cpu = seq_lens_cpu.int()
            actual_q = torch.cumsum(seq_lens_cpu, dim=0).int().to(device)
            fm.actual_seq_lengths_q = actual_q
            fm.actual_seq_lengths_q_pa = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                dim=0,
            )
        elif forward_batch.forward_mode.is_decode():
            B = forward_batch.batch_size
            fm.actual_seq_lengths_q = torch.arange(
                1, B + 1, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q_pa = torch.arange(
                0, B + 1, dtype=torch.int32, device=device
            )
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            B = forward_batch.batch_size
            from sglang.srt.utils.common import get_global_server_args

            n_draft = get_global_server_args().speculative_num_draft_tokens or 1
            actual_q = torch.arange(
                n_draft, B * n_draft + 1, n_draft, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q = actual_q
            fm.actual_seq_lengths_q_pa = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                dim=0,
            )
        else:
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None

        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            local_q_lens_cpu, _ = nsa_cp_round_robin_split_q_seqs_cpu(
                forward_batch.extend_seq_lens_cpu,
                keep_zeros=True,
            )
            local_q_lens = torch.tensor(
                local_q_lens_cpu,
                dtype=torch.int32,
                device=device,
            )
            actual_q = local_q_lens.cumsum(0)
            fm.actual_seq_lengths_q = actual_q
            fm.actual_seq_lengths_q_pa = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                dim=0,
            )

        # SWA page table -- populated by AscendAttnBackend when the model is
        # hybrid-SWA, else None. Aliased under the name forward_sparse uses.
        # Use explicit `is not None` check (not `or`) because
        # `bool(multi-element tensor)` raises.
        block_tables_swa = getattr(fm, "block_tables_swa", None)
        fm.swa_page_table = (
            block_tables_swa if block_tables_swa is not None else fm.block_tables
        )

        # actual_seq_lengths_kv defaults to None on main; the V4 metadata
        # kernel needs an int32 device tensor of per-request KV lengths.
        if fm.actual_seq_lengths_kv is None:
            if fm.seq_lens_cpu_int is not None:
                fm.actual_seq_lengths_kv = fm.seq_lens_cpu_int.to(
                    device=forward_batch.seq_lens.device, dtype=torch.int32
                )
            else:
                fm.actual_seq_lengths_kv = forward_batch.seq_lens.to(torch.int32)

        # Build kernel_metadata dict. For V4-Flash we mainly need c1a (no
        # compress KV) right now; c4a/c128a follow when we add those paths.
        fm.kernel_metadata = self._compute_kernel_metadata(forward_batch)
        self._ensure_cp_prefill_row_kernel_metadata(forward_batch)
        self._ensure_cp_prefill_batch_kernel_metadata(forward_batch)

        # Step-3 NPU compress metadata: only built when forward_npu paths are
        # active (env-gated). Each field is a per-request tensor consumed by
        # dsv4/{compressor,indexer}.py forward_npu. See iforgetmyname/dsv4_
        # release ascend_backend.init_forward_metadata @ ~L735-790 for the
        # reference impl on top of pre-allocated req_to_token_c{N} tables;
        # main has no req_to_token_c{N}, so we compute equivalents on the
        # fly from req_to_token + the V4 KV pool's swa translation.

        if envs.SGLANG_DSV4_NPU_REAL_COMPRESSOR.get() and self._dsv4_compress_ratios:
            self._build_npu_compress_metadata(forward_batch)

    def _compute_kernel_metadata(
        self,
        forward_batch: "ForwardBatch",
        *,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        override_max_seqlen_kv: Optional[int] = None,
    ) -> dict:
        fm = self.forward_metadata
        if batch_size is None:
            batch_size = forward_batch.batch_size
        # iforgetmyname Talantan1102/sglang#1: clamp seqused_kv >= 1 so idle
        # ranks (where actual_seq_lengths_kv may contain 0) don't trip
        # sparse-attn metadata with a zero-length entry, which has bitten the
        # NPU kernel in the dpattn + idle-rank workload.
        seqused_kv_safe = fm.actual_seq_lengths_kv.clamp(min=1)
        max_seqlen_kv = (
            int(seqused_kv_safe.max().item()) if seqused_kv_safe.numel() > 0 else 0
        )
        if max_seqlen_kv > 0:
            max_seqlen_kv = (
                (max_seqlen_kv + self.page_size - 1) // self.page_size
            ) * self.page_size
        if override_max_seqlen_kv is not None:
            max_seqlen_kv = max(max_seqlen_kv, int(override_max_seqlen_kv))
        if max_seqlen_q is None:
            cu_q = fm.actual_seq_lengths_q_pa
            if cu_q is not None and cu_q.numel() > 1:
                max_seqlen_q = int((cu_q[1:] - cu_q[:-1]).max().item())
            elif fm.actual_seq_lengths_q is not None and fm.actual_seq_lengths_q.numel() > 0:
                max_seqlen_q = int(fm.actual_seq_lengths_q.max().item())
            else:
                max_seqlen_q = 1
        max_seqlen_q = max(int(max_seqlen_q), 1)

        cu_seqlens_q = fm.actual_seq_lengths_q_pa
        seqused_kv = seqused_kv_safe
        if isinstance(cu_seqlens_q, torch.Tensor):
            cu_seqlens_q = torch.tensor(
                cu_seqlens_q.detach().cpu().tolist(),
                dtype=torch.int32,
                device=cu_seqlens_q.device,
            )
        if isinstance(seqused_kv, torch.Tensor):
            seqused_kv = torch.tensor(
                seqused_kv.detach().cpu().tolist(),
                dtype=torch.int32,
                device=seqused_kv.device,
            )

        common = {
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_kv": seqused_kv,
            "cmp_ratio": 1,
            "ori_mask_mode": 4,  # sliding window
            "cmp_mask_mode": 3,  # causal
            "ori_win_left": self._dsv4_sliding_window_size - 1,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        }
        base_kwargs = {
            "batch_size": batch_size,
            "num_heads_q": self._dsv4_q_head_num,
            "num_heads_kv": self._dsv4_kv_head_num,
            "head_dim": self._dsv4_head_dim,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "has_ori_kv": True,
            "has_cmp_kv": False,
        }
        c1a_kwargs = base_kwargs | common
        kernel_metadata = {
            "c1a_metadata": torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                **c1a_kwargs
            )
        }

        if self._dsv4_has_c4:
            c4a_overrides = {
                "cmp_ratio": 4,
                "has_cmp_kv": True,
                "cmp_topk": self._dsv4_index_topk,
            }
            if max_seqlen_kv >= 4:
                c4a_kwargs = c1a_kwargs | c4a_overrides
                kernel_metadata["c4a_metadata"] = (
                    torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c4a_kwargs)
                )
            else:
                kernel_metadata["c4a_metadata"] = kernel_metadata["c1a_metadata"]

            # The lightning indexer is only attached to c4 layers.
            # Pass actual_seq_lengths_q (no leading 0, B-element cumsum)
            # exactly as iforgetmyname/dsv4_release builds it — a fresh
            # contiguous int32 device tensor, not a slice.
            actual_q = fm.actual_seq_lengths_q
            if actual_q is None:
                actual_q = fm.actual_seq_lengths_kv
            kernel_metadata["li_quant_metadata"] = (
                torch.ops.custom.npu_quant_lightning_indexer_metadata(
                    device=str(actual_q.device),
                    actual_seq_lengths_query=actual_q,
                    actual_seq_lengths_key=fm.actual_seq_lengths_kv,
                    layout_key="PA_BSND",
                    sparse_count=self._dsv4_index_topk,
                    sparse_mode=3,
                    layout_query="TND",
                    cmp_ratio=4,
                    key_quant_mode=0,
                    query_quant_mode=0,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                )
            )

        if self._dsv4_has_c128:
            c128a_overrides = {"cmp_ratio": 128, "has_cmp_kv": True}
            if max_seqlen_kv >= 128:
                c128a_kwargs = c1a_kwargs | c128a_overrides
                kernel_metadata["c128a_metadata"] = (
                    torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c128a_kwargs)
                )
            else:
                kernel_metadata["c128a_metadata"] = kernel_metadata["c1a_metadata"]

        if self._dsv4_has_c4 and max_seqlen_kv >= 4:
            _ = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c1a_kwargs)

        return kernel_metadata

    def _clone_kernel_metadata(self, kernel_metadata: dict) -> dict:
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in kernel_metadata.items()
        }

    def _dsv4_cp_round_robin_local_positions(
        self,
        forward_batch: "ForwardBatch",
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = getattr(forward_batch, "positions", None)
        if positions is not None and positions.numel() > 0:
            if positions.numel() == num_tokens:
                local_positions = positions
            else:
                local_positions = nsa_cp_round_robin_split_data(positions)
            if local_positions.numel() >= num_tokens:
                return local_positions[:num_tokens].to(
                    device=device,
                    dtype=torch.int64,
                )

        # Fallback for metadata-only callers: reconstruct the same round-robin
        # positions from the local query row count.
        from sglang.srt.layers.attention.nsa.utils import (
            get_nsa_prefill_cp_rank,
            get_nsa_prefill_cp_size,
        )

        cp_rank = get_nsa_prefill_cp_rank()
        cp_size = get_nsa_prefill_cp_size()
        return torch.arange(
            cp_rank,
            cp_rank + num_tokens * cp_size,
            cp_size,
            device=device,
            dtype=torch.int64,
        )

    def _compute_single_query_kernel_metadata(
        self,
        forward_batch: "ForwardBatch",
        seqused_kv: int,
        global_q_pos: int = 0,
    ) -> dict:
        """Build kernel metadata for a single-Q-token batch entry.

        ``global_q_pos`` is the absolute position of this Q token in the full
        context (0-based).  The single-row kernel call still uses local
        ``cu_seqlens_q=[0, 1]`` because the q tensor passed to
        ``npu_sparse_attn_sharedkv`` has exactly one row.  The global causal KV
        boundary is expressed through ``seqused_kv``.
        """
        _ = global_q_pos  # Kept in the signature to document the caller context.
        fm = self.forward_metadata
        saved_actual_q = getattr(fm, "actual_seq_lengths_q", None)
        saved_actual_q_pa = getattr(fm, "actual_seq_lengths_q_pa", None)
        saved_actual_q_cmp = getattr(fm, "actual_seq_lengths_q_cmp", None)
        saved_actual_kv = getattr(fm, "actual_seq_lengths_kv", None)
        q_device = (
            saved_actual_q_pa.device
            if isinstance(saved_actual_q_pa, torch.Tensor)
            else forward_batch.seq_lens.device
        )
        kv_device = (
            saved_actual_kv.device
            if isinstance(saved_actual_kv, torch.Tensor)
            else q_device
        )
        try:
            fm.actual_seq_lengths_q = torch.tensor(
                [1],
                device=q_device,
                dtype=torch.int32,
            )
            # cu_seqlens_q indexes the one-row q tensor used for row metadata.
            # The global causal/KV boundary is carried by seqused_kv.
            fm.actual_seq_lengths_q_pa = torch.tensor(
                [0, 1],
                device=q_device,
                dtype=torch.int32,
            )
            fm.actual_seq_lengths_q_cmp = fm.actual_seq_lengths_q_pa.clone()
            fm.actual_seq_lengths_kv = torch.tensor(
                [max(1, int(seqused_kv))],
                device=kv_device,
                dtype=torch.int32,
            )
            return self._compute_kernel_metadata(forward_batch, batch_size=1)
        finally:
            fm.actual_seq_lengths_q = saved_actual_q
            fm.actual_seq_lengths_q_pa = saved_actual_q_pa
            fm.actual_seq_lengths_q_cmp = saved_actual_q_cmp
            fm.actual_seq_lengths_kv = saved_actual_kv

    def _ensure_cp_prefill_row_kernel_metadata(
        self,
        forward_batch: "ForwardBatch",
    ) -> None:
        self.dsv4_cp_prefill_row_kernel_metadata = {}
        if not can_nsa_prefill_cp_round_robin_split(forward_batch):
            return
        if get_bool_env_var("SGLANG_DSV4_NPU_DISABLE_PREFILL_ROW_METADATA", "False"):
            return

        fm = self.forward_metadata
        actual_q_pa = getattr(fm, "actual_seq_lengths_q_pa", None)
        if actual_q_pa is None or actual_q_pa.numel() == 0:
            return
        q_rows = int(actual_q_pa[-1].item())
        if q_rows <= 0:
            return

        local_positions = self._dsv4_cp_round_robin_local_positions(
            forward_batch,
            q_rows,
            actual_q_pa.device,
        )
        # For round-robin CP each local position is unique, so seqused_kv
        # (= local_pos + 1) is a bijection to the row index.  Key metadata by
        # seqused_kv; pass global_q_pos = local_pos so the baked
        # cu_seqlens_q = [local_pos, local_pos+1] encodes the absolute
        # position — without this the SWA/causal mask always places Q at
        # position 0, restricting every row to attend only to KV[0].
        local_positions_list = local_positions.detach().cpu().tolist()
        unique_seqused_kv = []
        for local_pos in local_positions_list:
            local_pos = int(local_pos)
            seq_len = local_pos + 1  # seqused_kv for this row
            if seq_len in self.dsv4_cp_prefill_row_kernel_metadata:
                continue  # already pre-warmed (duplicate global position)
            unique_seqused_kv.append(seq_len)
            self.dsv4_cp_prefill_row_kernel_metadata[seq_len] = (
                self._clone_kernel_metadata(
                    self._compute_single_query_kernel_metadata(
                        forward_batch,
                        seqused_kv=seq_len,
                        global_q_pos=local_pos,
                    )
                )
            )
        if get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False"):
            logger.warning(
                "DSV4 NPU warmed CP prefill row metadata: q_rows=%s, "
                "unique_rows=%s, min_seqused=%s, max_seqused=%s, positions=%s",
                q_rows,
                len(unique_seqused_kv),
                min(unique_seqused_kv) if len(unique_seqused_kv) > 0 else None,
                max(unique_seqused_kv) if len(unique_seqused_kv) > 0 else None,
                local_positions_list,
            )

    def _ensure_cp_prefill_batch_kernel_metadata(
        self,
        forward_batch: "ForwardBatch",
    ) -> None:
        """Retained as a no-op placeholder.

        The batched virtual-request approach (one npu_sparse_attn_sharedkv call
        with batch_size=n_local and per-element seqused_kv) caused aicore DDR
        errors (errcode 507015) when multiple virtual requests share the same
        physical KV page — the NPU kernel does not support shared-page batched
        calls.  The forward methods now use the per-row loop backed by
        _ensure_cp_prefill_row_kernel_metadata instead.  This stub preserves
        the attribute so callers that reference dsv4_cp_prefill_batch_kernel_metadata
        don't raise AttributeError.
        """
        self.dsv4_cp_prefill_batch_kernel_metadata = {}

    def _build_npu_compress_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Populate c{4,128}_{page_table,state_page_table,state_loc,loc} on
        forward_metadata for the NPU compressor / indexer forward_npu paths.

        Reference: iforgetmyname/dsv4_release ascend_backend.init_forward_metadata
        @ ~L735-790. iforgetmyname pre-allocates per-request mapping tables
        (req_to_token_c4 / req_to_token_c4_state) when the request enters the
        scheduler; main has no such tables, so we compute equivalents on the
        fly from req_to_token + the V4 KV pool's swa translation. This is
        slower but avoids cross-cutting allocator surgery on the request pool.
        """
        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool = forward_batch.req_pool_indices
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        is_decode = forward_batch.forward_mode.is_decode()

        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_max = int(seq_lens.max().item()) if bs > 0 else 0
        n_pages = max(1, (seq_lens_max + self.page_size - 1) // self.page_size)

        # State page tables — for each request, for each page, the state-buffer
        # page index. Use the FIRST token of each page as the representative
        # (tokens within the same SWA page produce contiguous state-buffer slots).
        page_starts = torch.arange(
            0, n_pages * self.page_size, self.page_size, device=device
        )  # [n_pages]
        # [bs, n_pages] flattened token positions; positions past seq_len are
        # clamped to 0 (will be masked out by _get_kv_indices' kv_len).
        page_starts_2d = page_starts.unsqueeze(0).expand(bs, n_pages)
        # Index req_to_token: [bs, n_pages] of full-kv-pool slot ids.
        raw_loc = req_to_token[
            req_pool.unsqueeze(1).expand(-1, n_pages), page_starts_2d
        ]

        for ratio in self._dsv4_compress_ratios:
            if ratio not in (4, 128):
                continue
            # State page table — translate each (bs, n_pages) raw kv slot to a
            # state-buffer page id. translate_kv_loc_to_compress_state_loc gives
            # the flat state slot; divide by page_size for the page id.
            state_loc_2d = pool.translate_kv_loc_to_compress_state_loc(raw_loc, ratio)
            state_page_2d = (state_loc_2d // self.page_size).to(torch.int32)

            # State loc — single state-buffer slot for the new decode token.
            # In decode, out_cache_loc has shape [bs] (one new token per req).
            if is_decode:
                state_loc_decode = pool.translate_kv_loc_to_compress_state_loc(
                    forward_batch.out_cache_loc, ratio
                )
                # Compressor write loc — step 5c slab allocator. For each
                # request that just completed a ratio-aligned chunk, the new
                # compressed token writes to slot
                #   k_seq = seqlen_after // ratio - 1     (compressed seq pos)
                #   slot  = req_to_token_c{N}_pages[req_pool_idx, k_seq // page_size]
                #           * page_size + k_seq % page_size
                # Replaces the old `raw_out_loc // ratio` formula which only
                # worked when the request happened to land on a page-aligned
                # raw kv slot (= almost never).
                pages_table = pool.get_req_to_token_c_pages(ratio)
                should_compress = (seq_lens % ratio) == 0
                k_seq = (seq_lens.to(torch.int64) // ratio - 1).clamp(min=0)
                page_seq = (k_seq // self.page_size).to(torch.int64)
                offset = (k_seq % self.page_size).to(torch.int64)
                kernel_page = pages_table[req_pool.to(torch.int64), page_seq].to(
                    torch.int64
                )
                compress_out_loc = (kernel_page * self.page_size + offset).to(
                    torch.int32
                )
                compress_out_loc = torch.where(
                    should_compress,
                    compress_out_loc,
                    torch.zeros_like(compress_out_loc),
                )
            else:
                state_loc_decode = None
                compress_out_loc = None

            attr_state_pt = f"c{ratio}_state_page_table"
            attr_state_loc = f"c{ratio}_state_loc"
            attr_loc = f"c{ratio}_loc"
            setattr(fm, attr_state_pt, state_page_2d)
            setattr(fm, attr_state_loc, state_loc_decode)
            setattr(fm, attr_loc, compress_out_loc)

            # c{ratio}_page_table — kernel-view page table for c{N}_kv_pool.
            # Step 5c: read directly from the slab — gives each request its
            # own dedicated kernel pages so cmp_kv reads at compressed seq
            # pos 0..N-1 land in the right physical slots regardless of how
            # the raw_kv allocator scattered the request's full pages.
            pages_table = pool.get_req_to_token_c_pages(ratio)
            n_pages_c = (n_pages + ratio - 1) // ratio
            n_pages_c = max(1, min(n_pages_c, pages_table.shape[1]))
            c_page_table = pages_table[req_pool.to(torch.int64), :n_pages_c].to(
                torch.int32
            )
            setattr(fm, f"c{ratio}_page_table", c_page_table)

    def init_forward_metadata_indexer(self, core_attn_metadata):
        # li_quant_metadata is computed inside _compute_kernel_metadata; nothing
        # extra to do here. Return None to satisfy the mixin contract.
        return None

    def _seed_c4_topk_indices(self, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Allocate a [T, index_topk] int32 tensor on the compute device,
        filled with -1 (= "no valid sparse index" sentinel that npu_sparse_
        attn_sharedkv accepts). Real ``forward_c4_indexer`` will overwrite the
        contents via ``npu_quant_lightning_indexer``; until then this lets the
        c4 path of ``_forward_compressed`` consume a well-shaped tensor."""
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            T = int(self.forward_metadata.actual_seq_lengths_q_pa[-1].item())
        elif forward_batch.input_ids is not None:
            T = forward_batch.input_ids.shape[0]
        else:
            T = int(forward_batch.seq_lens.sum().item())
        return torch.full(
            (T, self._dsv4_index_topk),
            -1,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        *,
        compress_ratio: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if compress_ratio not in (0, 1, 4, 128):
            raise ValueError(
                f"V4 attention expects compress_ratio in (0, 1, 4, 128); got {compress_ratio}"
            )
        # IDLE rank short-circuit: DP-attention pads idle ranks (bs>0,
        # seq_lens all zero) so collective ops stay synchronized. The output
        # is thrown away by the DP allreduce, but if we still execute the
        # NPU sparse-attn / store_cache kernels the AICPU path bombs out
        # (kernelName=SparseAttnSharedkvMetadata, errcode 0x2a) the next
        # time a sync point is reached. Mirror the pattern used by
        # forward_compress / forward_c4_indexer above: skip every NPU custom
        # op for idle and return zeros of the expected attention-output
        # shape so the rest of model.forward can run for collective sync.
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(q)
        # Honor save_kv_cache=True contract. With SGLANG_OPT_USE_OVERLAP_STORE_CACHE
        # default TRUE, MQALayer._forward_prepare already writes K via store_cache
        # and passes save_kv_cache=False here (no dup-write). With overlap=False,
        # the previous code silently dropped the write — decode then read an
        # unwritten swa_kv_pool and produced garbage. Always respect the flag.
        if save_kv_cache:
            self.store_cache(
                layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch
            )
        if compress_ratio in (0, 1):
            return self._forward_dense(
                q, k, layer, forward_batch, attn_sink, compress_ratio
            )
        # ratio 4 / 128 routing — TWO independent gates:
        #   SGLANG_DSV4_NPU_REAL_COMPRESSOR=1 turns on the in-module
        #     forward_npu (compressor writes real KV; output unchanged
        #     because attention still falls back to dense here).
        #   SGLANG_DSV4_NPU_SPARSE_ATTN=1 additionally routes attention
        #     through _forward_compressed (has_cmp_kv=True kernel path).
        # The second gate stays OFF by default until the kernel call's
        # size / sparse-indices mismatch is resolved; with it OFF, output
        # is bit-for-bit identical to the flag-OFF baseline.

        sparse_on = envs.SGLANG_DSV4_NPU_SPARSE_ATTN.get()
        c128_only = envs.SGLANG_DSV4_NPU_SPARSE_ATTN_C128_ONLY.get()
        # Bisect mode: only c128 layers route to _forward_compressed.
        if c128_only and compress_ratio != 128:
            return self._forward_dense(
                q, k, layer, forward_batch, attn_sink, compress_ratio
            )
        if sparse_on or c128_only:
            return self._forward_compressed(
                q, k, layer, forward_batch, attn_sink, compress_ratio
            )
        return self._forward_dense(
            q, k, layer, forward_batch, attn_sink, compress_ratio
        )

    def _build_cp_prefill_contiguous_swa_inputs(
        self,
        k: torch.Tensor,
        forward_batch: "ForwardBatch",
        default_page_table: Optional[torch.Tensor],
        layer_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Build the PA_ND inputs used by the original V4 NPU CP prefill path.

        In round-robin CP prefill, ``k`` has already been all-gathered and
        reranged back to global token order while ``q`` stays local. Reading
        through the global SWA pool/page table can observe allocator-specific
        layout differences; the original branch instead uses this gathered K
        as a compact per-request page buffer and a matching contiguous block
        table. This helper mirrors that contract for the A-stage dense path.
        """
        if k is None or k.numel() == 0:
            return None, None

        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            return None, None
        if isinstance(seq_lens_cpu, torch.Tensor):
            seq_lens_list = [int(x) for x in seq_lens_cpu.cpu().tolist()]
        else:
            seq_lens_list = [int(x) for x in seq_lens_cpu]
        total_seq_len = sum(seq_lens_list)
        if total_seq_len <= 0:
            return None, None

        expected_k_rows = get_nsa_prefill_cp_total_len(forward_batch)
        if k.shape[0] != total_seq_len or expected_k_rows != total_seq_len:
            if (
                get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False")
                and (
                    layer_id is None
                    or _dsv4_npu_should_log_layer(
                        layer_id, self._dsv4_num_hidden_layers
                    )
                )
            ):
                logger.warning(
                    "DSV4 NPU CP keeps pool SWA inputs because gathered K does "
                    "not cover the full context: k_rows=%s, total_seq_len=%s, "
                    "expected_cp_total=%s, seq_lens=%s, extend_lens=%s",
                    k.shape[0],
                    total_seq_len,
                    expected_k_rows,
                    seq_lens_list,
                    getattr(forward_batch, "extend_seq_lens_cpu", None),
                )
            return None, None

        k_rows = k.unsqueeze(1) if k.ndim == 2 else k
        if k_rows.ndim != 3:
            raise RuntimeError(
                "DeepSeek V4 CP prefill expects gathered K with shape "
                f"(T, D) or (T, 1, D), got {tuple(k.shape)}."
            )

        page_size = self.page_size
        num_pages_list = [
            (seq_len + page_size - 1) // page_size for seq_len in seq_lens_list
        ]
        total_pages = sum(num_pages_list)
        if total_pages <= 0:
            return None, None

        kv_pad = k_rows.new_zeros(total_pages * page_size, *k_rows.shape[1:])
        src_offset = 0
        page_offset = 0
        for seq_len, num_pages in zip(seq_lens_list, num_pages_list):
            if seq_len > 0:
                dst_start = page_offset * page_size
                kv_pad[dst_start : dst_start + seq_len].copy_(
                    k_rows[src_offset : src_offset + seq_len]
                )
                src_offset += seq_len
            page_offset += num_pages
        ori_kv = kv_pad.view(total_pages, page_size, *k_rows.shape[1:])

        if default_page_table is not None:
            block_device = default_page_table.device
            block_dtype = default_page_table.dtype
            max_blocks = default_page_table.shape[1]
        else:
            block_device = k.device
            block_dtype = torch.int32
            max_blocks = max(num_pages_list, default=1)

        num_pages_tensor = torch.tensor(
            num_pages_list, dtype=block_dtype, device=block_device
        )
        page_offsets = torch.zeros(
            len(num_pages_list), dtype=block_dtype, device=block_device
        )
        if len(num_pages_list) > 1:
            page_offsets[1:] = num_pages_tensor[:-1].cumsum(0)
        arange = torch.arange(max_blocks, dtype=block_dtype, device=block_device)
        ori_block_table = page_offsets.unsqueeze(1) + arange.unsqueeze(0)
        ori_block_table = torch.where(
            arange.unsqueeze(0) < num_pages_tensor.unsqueeze(1),
            ori_block_table,
            torch.zeros_like(ori_block_table),
        ).contiguous()

        if (
            get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False")
            and (
                layer_id is None
                or _dsv4_npu_should_log_layer(layer_id, self._dsv4_num_hidden_layers)
            )
        ):
            logger.warning(
                "DSV4 NPU CP uses contiguous gathered K for SWA prefill: "
                "k_rows=%s, ori_kv_shape=%s, ori_block_table_shape=%s, "
                "seq_lens=%s",
                k.shape[0],
                tuple(ori_kv.shape),
                tuple(ori_block_table.shape),
                seq_lens_list,
            )
        return ori_kv, ori_block_table

    def _dsv4_cp_prefill_local_batch_indices(
        self,
        forward_batch: "ForwardBatch",
        num_tokens: int,
    ) -> Optional[list[int]]:
        local_q_lens_cpu, _ = nsa_cp_round_robin_split_q_seqs_cpu(
            forward_batch.extend_seq_lens_cpu,
            keep_zeros=True,
        )
        batch_indices: list[int] = []
        for batch_idx, q_len in enumerate(local_q_lens_cpu):
            batch_indices.extend([batch_idx] * int(q_len))
        if len(batch_indices) < num_tokens:
            return None
        return batch_indices[:num_tokens]

    def _forward_dense_cp_prefill_torch(
        self,
        q: torch.Tensor,
        ori_kv: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        local_positions: torch.Tensor,
        local_seqused_kv: torch.Tensor,
        row_batch_indices: Optional[list[int]],
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """Slow reference path for short CP prefill precision bisection."""
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            raise RuntimeError("Torch dense CP prefill requires seq_lens_cpu.")
        if isinstance(seq_lens_cpu, torch.Tensor):
            seq_lens_list = [int(x) for x in seq_lens_cpu.cpu().tolist()]
        else:
            seq_lens_list = [int(x) for x in seq_lens_cpu]

        num_pages_list = [
            (seq_len + self.page_size - 1) // self.page_size
            for seq_len in seq_lens_list
        ]
        page_offsets: list[int] = []
        acc = 0
        for num_pages in num_pages_list:
            page_offsets.append(acc)
            acc += num_pages

        kv_rows = ori_kv.reshape(-1, *ori_kv.shape[2:])
        if kv_rows.ndim != 3 or kv_rows.shape[1] != 1:
            raise RuntimeError(
                "Torch dense CP prefill expects PA_ND KV shaped "
                f"(pages, page_size, 1, dim), got {tuple(ori_kv.shape)}."
            )

        if q.ndim != 3:
            raise RuntimeError(
                f"Torch dense CP prefill expects q shaped (T, H, D), got {tuple(q.shape)}."
            )

        sink = attn_sink.detach().float() if attn_sink is not None else None
        out = torch.empty_like(q)
        for row_idx in range(q.shape[0]):
            batch_idx = (
                row_batch_indices[row_idx]
                if row_batch_indices is not None
                else 0
            )
            row_seq_len = int(local_seqused_kv[row_idx].item())
            row_seq_len = min(row_seq_len, seq_lens_list[batch_idx])
            win_start = max(0, row_seq_len - self._dsv4_sliding_window_size)
            base = page_offsets[batch_idx] * self.page_size
            k_visible = kv_rows[
                base + win_start : base + row_seq_len,
                0,
                :,
            ].float()
            if k_visible.numel() == 0:
                out[row_idx].zero_()
                continue

            q_row = q[row_idx].float()
            scores = torch.matmul(q_row, k_visible.transpose(0, 1)) * layer.scaling
            if sink is not None:
                sink_logits = sink.to(device=scores.device).reshape(-1, 1)
                scores = torch.cat([scores, sink_logits], dim=-1)
                probs = torch.softmax(scores, dim=-1)[..., :-1]
            else:
                probs = torch.softmax(scores, dim=-1)
            out[row_idx].copy_(torch.matmul(probs, k_visible).to(dtype=q.dtype))

        if (
            get_bool_env_var("SGLANG_DSV4_NPU_CP_VERIFY", "False")
            and _dsv4_npu_should_log_layer(
                layer.layer_id, self._dsv4_num_hidden_layers
            )
        ):
            logger.warning(
                "DSV4 CP torch dense prefill: layer=%s ratio=%s n_local=%s "
                "q_norm=%.4f out_norm=%.4f out_max=%.4f positions=%s",
                layer.layer_id,
                compress_ratio,
                q.shape[0],
                float(q.float().norm().item()),
                float(out.float().norm().item()),
                float(out.float().abs().max().item()),
                local_positions.detach().cpu().tolist(),
            )
        return out

    def _forward_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """ratio=1 / ratio=0 dense layers — sliding-window attention via
        npu_sparse_attn_sharedkv with has_cmp_kv=False."""
        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        ori_kv = pool.get_swa_buffer(layer.layer_id)  # (num_pages, page_size, 1, dim)
        ori_block_table = fm.swa_page_table
        attn_sink = _dsv4_prepare_attn_sink(
            attn_sink, layer.layer_id, self._dsv4_num_hidden_layers
        )
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            cp_ori_kv, cp_ori_block_table = self._build_cp_prefill_contiguous_swa_inputs(
                k,
                forward_batch,
                fm.swa_page_table,
                layer.layer_id,
            )
            if cp_ori_kv is not None:
                ori_kv = cp_ori_kv
                ori_block_table = cp_ori_block_table

        attn_kwargs = dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_kv=fm.actual_seq_lengths_kv,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=ori_kv,
            ori_block_table=ori_block_table,
            sinks=attn_sink,
            metadata=fm.kernel_metadata["c1a_metadata"],
            softmax_scale=layer.scaling,
        )
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            if q.shape[0] == 0:
                return q
            n_local = q.shape[0]
            local_positions = self._dsv4_cp_round_robin_local_positions(
                forward_batch,
                n_local,
                q.device,
            )
            # Each local row r has global causal boundary cp_rank + r*cp_size.
            # Issue one kernel call per Q row so each call carries the correct
            # seqused_kv for that specific row.  Per-row metadata is pre-warmed
            # in _ensure_cp_prefill_row_kernel_metadata; the fallback uses the
            # main fm.kernel_metadata (batch_size=1, matches per-row call).
            local_seqused_kv = (local_positions + 1).to(fm.actual_seq_lengths_kv.dtype)
            # cu_seqlens_q_dtype — reuse from the forward-metadata tensor.
            _cu_q_dtype = fm.actual_seq_lengths_q_pa.dtype
            row_batch_indices = self._dsv4_cp_prefill_local_batch_indices(
                forward_batch,
                n_local,
            )
            _cp_verify = get_bool_env_var(
                "SGLANG_DSV4_NPU_CP_VERIFY", "False"
            ) and _dsv4_npu_should_log_layer(
                layer.layer_id, self._dsv4_num_hidden_layers
            )
            _torch_dense_prefill = get_bool_env_var(
                "SGLANG_DSV4_NPU_TORCH_DENSE_PREFILL", "False"
            )
            _torch_dense_compressed_only = get_bool_env_var(
                "SGLANG_DSV4_NPU_TORCH_DENSE_PREFILL_COMPRESSED_ONLY", "False"
            )
            if _torch_dense_prefill and (
                not _torch_dense_compressed_only or compress_ratio in (4, 128)
            ):
                return self._forward_dense_cp_prefill_torch(
                    q,
                    ori_kv,
                    layer,
                    forward_batch,
                    local_positions,
                    local_seqused_kv,
                    row_batch_indices,
                    attn_sink,
                    compress_ratio,
                )
            _skip_row_meta = get_bool_env_var(
                "SGLANG_DSV4_NPU_CP_SKIP_ROW_META", "False"
            )
            if _cp_verify:
                logger.warning(
                    "DSV4 CP NPU dense prefill: layer=%s ratio=%s n_local=%s "
                    "ori_kv_shape=%s ori_block_table_shape=%s "
                    "q_norm=%.4f seqused_kv_min=%s seqused_kv_max=%s",
                    layer.layer_id,
                    compress_ratio,
                    n_local,
                    tuple(ori_kv.shape),
                    tuple(ori_block_table.shape),
                    float(q.float().norm().item()),
                    int(local_seqused_kv.min().item()),
                    int(local_seqused_kv.max().item()),
                )
            out = torch.empty_like(q)
            for row_idx in range(n_local):
                row_seq_len = int(local_seqused_kv[row_idx].item())
                row_global_q_pos = int(local_positions[row_idx].item())
                batch_idx = (
                    row_batch_indices[row_idx]
                    if row_batch_indices is not None
                    else 0
                )
                # cu_seqlens_q indexes q[row_idx:row_idx+1]; use local offsets.
                # The global causal/KV boundary is carried by row_seqused_kv.
                row_cu_seqlens_q = torch.tensor(
                    [0, 1],
                    dtype=_cu_q_dtype,
                    device=q.device,
                )
                row_seqused_kv = torch.tensor(
                    [row_seq_len],
                    dtype=fm.actual_seq_lengths_kv.dtype,
                    device=q.device,
                )
                if _skip_row_meta:
                    # Use the main (full-batch) metadata for all rows.
                    # Diagnostic path: if this fixes wrong output the bug is
                    # in _ensure_cp_prefill_row_kernel_metadata.
                    row_metadata = fm.kernel_metadata["c1a_metadata"]
                else:
                    row_meta = self.dsv4_cp_prefill_row_kernel_metadata.get(
                        row_seq_len
                    )
                    row_metadata = (
                        row_meta["c1a_metadata"]
                        if row_meta
                        else fm.kernel_metadata["c1a_metadata"]
                    )
                row_attn_kwargs = dict(attn_kwargs)
                row_attn_kwargs.update(
                    {
                        "cu_seqlens_q": row_cu_seqlens_q,
                        "seqused_kv": row_seqused_kv,
                        "q": q[row_idx : row_idx + 1],
                        "ori_block_table": ori_block_table[
                            batch_idx : batch_idx + 1
                        ],
                        "metadata": row_metadata,
                    }
                )
                row_out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
                    **row_attn_kwargs
                )
                if _cp_verify and (
                    row_idx == 0
                    or row_idx == n_local - 1
                    or row_idx % max(1, n_local // 4) == 0
                ):
                    logger.warning(
                        "DSV4 CP NPU dense row: layer=%s row=%d/%d "
                        "global_q_pos=%d seqused_kv=%d batch_idx=%d "
                        "q_norm=%.4f out_norm=%.4f out_max=%.4f "
                        "out_first=%s",
                        layer.layer_id,
                        row_idx,
                        n_local,
                        row_global_q_pos,
                        row_seq_len,
                        batch_idx,
                        float(q[row_idx].float().norm().item()),
                        float(row_out.float().norm().item()),
                        float(row_out.float().abs().max().item()),
                        row_out.view(-1)[:4].float().tolist(),
                    )
                out[row_idx : row_idx + 1].copy_(row_out)
            return out

        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def _forward_compressed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """ratio=4 / ratio=128 layers — sliding-window + compressed-KV
        sparse attention via npu_sparse_attn_sharedkv with has_cmp_kv=True.

        cmp_kv (compressed KV) is read from the c4 / c128 pool buffer,
        which is currently zeros (compressor write path is still stubbed),
        so the compressed contribution to the output is zero. cmp_sparse_
        indices for c4 comes from forward_metadata.c4_topk_indices, which
        forward_c4_indexer currently seeds with -1 (= no valid sparse
        index) for the same reason. The point of this commit is to validate
        the kernel-call shape/dtype contract end-to-end before we land the
        compressor + indexer compute paths.
        """
        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        metadata = fm.kernel_metadata.get(f"c{compress_ratio}a_metadata")
        cmp_kv = pool.get_compress_buffer(layer.layer_id, False)
        attn_sink = _dsv4_prepare_attn_sink(
            attn_sink, layer.layer_id, self._dsv4_num_hidden_layers
        )

        if metadata is None or cmp_kv is None:
            raise RuntimeError(
                "DeepseekV4AscendAttnBackend._forward_compressed: missing "
                f"required state for layer_id={layer.layer_id} "
                f"compress_ratio={compress_ratio}. "
                f"metadata({'present' if metadata is not None else 'MISSING'}), "
                f"cmp_kv({'present' if cmp_kv is not None else 'MISSING'}). "
                f"Available kernel_metadata keys: {list(fm.kernel_metadata.keys())}. "
                "This indicates a configuration / pool-init bug — silently "
                "returning zeros would corrupt model output."
            )

        ori_kv = pool.get_swa_buffer(layer.layer_id)
        ori_block_table = fm.swa_page_table
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            cp_ori_kv, cp_ori_block_table = self._build_cp_prefill_contiguous_swa_inputs(
                k,
                forward_batch,
                fm.swa_page_table,
                layer.layer_id,
            )
            if cp_ori_kv is not None:
                ori_kv = cp_ori_kv
                ori_block_table = cp_ori_block_table

        # Reshape cmp_kv to share page_size with ori_kv before the kernel call.
        # main's V4 pool layout: c{N}_kv_pool buffer is (num_pages, page_size//
        # ratio, 1, dim) so each native page holds page_size//ratio compressed
        # tokens. The aclnn kernel expects cmp_kv to share its page_size with
        # ori_kv (=global page_size). We slice the buffer to a ratio-aligned
        # native-page count and view it as (N_kernel, global_page_size, 1, dim).
        #
        # cmp_block_table values: step 5c slab (`req_to_token_c{N}_pages`)
        # already gives kernel-view page indices in [0, N_kernel), so no
        # further `// page_ratio` divide is needed — the divide was a leftover
        # from step 5b when block_table came from raw kv pool page indices.
        ori_page_size = ori_kv.shape[1]
        cmp_native_page_size = cmp_kv.shape[1]
        cmp_block_table = getattr(
            fm, f"c{compress_ratio}_page_table", fm.swa_page_table
        )
        if cmp_native_page_size != ori_page_size:
            page_ratio = ori_page_size // cmp_native_page_size
            assert page_ratio == compress_ratio, (
                f"page_ratio={page_ratio} != compress_ratio={compress_ratio}; "
                "main's V4 pool keeps c{N}_native_page_size = global_page_size//ratio"
            )
            n_native = cmp_kv.shape[0]
            n_kernel = n_native // page_ratio
            cmp_kv = cmp_kv[: n_kernel * page_ratio].reshape(
                n_kernel, ori_page_size, *cmp_kv.shape[2:]
            )
            # Slab already in kernel-view page space — no divide.
            cmp_block_table = cmp_block_table.to(torch.int32)

        attn_kwargs = dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_kv=fm.actual_seq_lengths_kv,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=ori_kv,
            ori_block_table=ori_block_table,
            sinks=attn_sink,
            metadata=metadata,
            softmax_scale=layer.scaling,
            cmp_ratio=compress_ratio,
            cmp_mask_mode=3,
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
        )
        # Step-5c diagnosis: route c4 with cmp_sparse_indices=None (= same
        # treatment as c128) when SGLANG_DSV4_NPU_SPARSE_C4_NO_TOPK is set.
        # This bypasses the -1 sentinel topk path that was used to "mask"
        # all c4 history, and instead lets the kernel use the entire
        # populated c4 history (up to seqused_kv // ratio compressed
        # tokens). If output stabilizes after this, the divergence we see
        # in step-5c is due to the kernel mis-handling -1 in the c4 sparse
        # indices tensor, not due to slab/cmp_kv layout. If output still
        # diverges from dense baseline, the issue is in compressor write
        # values (ape/wkv split) or in lingering pool state.

        if compress_ratio == 4 and not envs.SGLANG_DSV4_NPU_SPARSE_C4_NO_TOPK.get():
            topk = fm.c4_topk_indices
            if topk is None:
                topk = self._seed_c4_topk_indices(forward_batch)
                fm.c4_topk_indices = topk
            attn_kwargs["cmp_sparse_indices"] = topk.view(-1, 1, topk.shape[-1])
        else:
            attn_kwargs["cmp_sparse_indices"] = None
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            if q.shape[0] == 0:
                return q
            n_local = q.shape[0]
            local_positions = self._dsv4_cp_round_robin_local_positions(
                forward_batch,
                n_local,
                q.device,
            )
            # Per-row kernel calls: each row has its own causal KV boundary.
            # Per-row metadata is pre-warmed in _ensure_cp_prefill_row_kernel_metadata.
            local_seqused_kv = (local_positions + 1).to(fm.actual_seq_lengths_kv.dtype)
            # cu_seqlens_q_dtype — reuse from the forward-metadata tensor.
            _cu_q_dtype = fm.actual_seq_lengths_q_pa.dtype
            row_batch_indices = self._dsv4_cp_prefill_local_batch_indices(
                forward_batch,
                n_local,
            )
            disable_prefill_cmp_kv = get_bool_env_var(
                "SGLANG_DSV4_NPU_DISABLE_PA_PREFILL_CMP_KV",
                "False",
            )
            _cp_verify = get_bool_env_var(
                "SGLANG_DSV4_NPU_CP_VERIFY", "False"
            ) and _dsv4_npu_should_log_layer(
                layer.layer_id, self._dsv4_num_hidden_layers
            )
            _skip_row_meta = get_bool_env_var(
                "SGLANG_DSV4_NPU_CP_SKIP_ROW_META", "False"
            )
            if _cp_verify:
                logger.warning(
                    "DSV4 CP compressed prefill: layer=%s ratio=%s n_local=%s "
                    "ori_kv_shape=%s ori_block_table_shape=%s "
                    "q_norm=%.4f seqused_kv_min=%s seqused_kv_max=%s",
                    layer.layer_id,
                    compress_ratio,
                    n_local,
                    tuple(ori_kv.shape),
                    tuple(ori_block_table.shape),
                    float(q.float().norm().item()),
                    int(local_seqused_kv.min().item()),
                    int(local_seqused_kv.max().item()),
                )
            topk_for_rows = attn_kwargs.get("cmp_sparse_indices")  # (bs, 1, topk_k) or None
            out = torch.empty_like(q)
            for row_idx in range(n_local):
                row_seq_len = int(local_seqused_kv[row_idx].item())
                row_global_q_pos = int(local_positions[row_idx].item())
                batch_idx = (
                    row_batch_indices[row_idx]
                    if row_batch_indices is not None
                    else 0
                )
                # cu_seqlens_q indexes q[row_idx:row_idx+1]; use local offsets.
                # The global causal/KV boundary is carried by row_seqused_kv.
                row_cu_seqlens_q = torch.tensor(
                    [0, 1],
                    dtype=_cu_q_dtype,
                    device=q.device,
                )
                row_seqused_kv = torch.tensor(
                    [row_seq_len],
                    dtype=fm.actual_seq_lengths_kv.dtype,
                    device=q.device,
                )
                # Determine whether this specific row has compressed KV.
                if disable_prefill_cmp_kv:
                    row_has_cmp = False
                elif compress_ratio == 4 and topk_for_rows is not None:
                    row_has_cmp = bool(
                        (topk_for_rows[batch_idx : batch_idx + 1] >= 0).any().item()
                    )
                else:
                    row_has_cmp = row_seq_len >= compress_ratio
                metadata_key = (
                    f"c{compress_ratio}a_metadata" if row_has_cmp else "c1a_metadata"
                )
                if _skip_row_meta:
                    # Diagnostic path: bypass per-row metadata, use the main
                    # (full-batch) metadata. If this fixes wrong output, the
                    # bug is in _ensure_cp_prefill_row_kernel_metadata.
                    row_metadata = fm.kernel_metadata.get(
                        metadata_key, fm.kernel_metadata["c1a_metadata"]
                    )
                else:
                    row_meta = self.dsv4_cp_prefill_row_kernel_metadata.get(row_seq_len)
                    if row_meta and metadata_key in row_meta:
                        row_metadata = row_meta[metadata_key]
                    elif row_meta:
                        row_metadata = row_meta.get(
                            "c1a_metadata", fm.kernel_metadata["c1a_metadata"]
                        )
                    else:
                        row_metadata = fm.kernel_metadata.get(
                            metadata_key, fm.kernel_metadata["c1a_metadata"]
                        )
                row_attn_kwargs = dict(attn_kwargs)
                row_attn_kwargs.update(
                    {
                        "cu_seqlens_q": row_cu_seqlens_q,
                        "seqused_kv": row_seqused_kv,
                        "q": q[row_idx : row_idx + 1],
                        "ori_block_table": ori_block_table[
                            batch_idx : batch_idx + 1
                        ],
                        "metadata": row_metadata,
                    }
                )
                if row_has_cmp:
                    # Per-row compressed block table.
                    if "cmp_block_table" in row_attn_kwargs:
                        row_attn_kwargs["cmp_block_table"] = row_attn_kwargs[
                            "cmp_block_table"
                        ][batch_idx : batch_idx + 1]
                    # Per-row topk indices: slice to (1, 1, topk_k).
                    if topk_for_rows is not None:
                        row_attn_kwargs["cmp_sparse_indices"] = topk_for_rows[
                            batch_idx : batch_idx + 1
                        ]
                else:
                    for key in (
                        "cmp_ratio",
                        "cmp_mask_mode",
                        "cmp_kv",
                        "cmp_sparse_indices",
                        "cmp_block_table",
                    ):
                        row_attn_kwargs.pop(key, None)
                row_out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
                    **row_attn_kwargs
                )
                if _cp_verify and (
                    row_idx == 0
                    or row_idx == n_local - 1
                    or row_idx % max(1, n_local // 4) == 0
                ):
                    logger.warning(
                        "DSV4 CP compressed row: layer=%s ratio=%s "
                        "row=%d/%d global_q_pos=%d seqused_kv=%d has_cmp=%s "
                        "q_norm=%.4f out_norm=%.4f out_max=%.4f",
                        layer.layer_id,
                        compress_ratio,
                        row_idx,
                        n_local,
                        row_global_q_pos,
                        row_seq_len,
                        row_has_cmp,
                        float(q[row_idx].float().norm().item()),
                        float(row_out.float().norm().item()),
                        float(row_out.float().abs().max().item()),
                    )
                out[row_idx : row_idx + 1].copy_(row_out)
            return out
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def store_cache(self, *, layer_id: int, swa_k: torch.Tensor, forward_batch):
        """Write the SWA layer's K cache into the bf16 PA_ND buffer.

        ``swa_k`` arrives shaped (T, num_kv_heads=1, dim) where dim packs
        K_nope + K_rope in bf16 (same layout as get_swa_buffer returns).
        ``forward_batch.out_cache_loc`` is in FULL-pool index space (size
        = sum of all KV pools); the swa_kv_pool buffer is its own smaller
        space. We must translate full→swa first — otherwise the
        index_put hits the wrong slot (or wraps OOB), and decode reads
        garbage K back. This mirrors what the CUDA radix path does at
        set_swa_key_buffer_radix.
        """
        pool = forward_batch.token_to_kv_pool
        swa_loc = pool.translate_loc_from_full_to_swa(forward_batch.out_cache_loc)
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            cache_rows = int(swa_k.shape[0])
            loc_rows = int(swa_loc.numel())
            if cache_rows < loc_rows:
                if get_bool_env_var(
                    "SGLANG_DSV4_NPU_CP_VERIFY", "False"
                ) and _dsv4_npu_should_log_layer(
                    layer_id, self._dsv4_num_hidden_layers
                ):
                    logger.warning(
                        "DSV4 NPU CP trims padded SWA cache locs before write: "
                        "layer_id=%s, cache_rows=%s, loc_rows=%s, "
                        "num_token_non_padded=%s, extend_lens=%s",
                        layer_id,
                        cache_rows,
                        loc_rows,
                        getattr(forward_batch, "num_token_non_padded_cpu", None),
                        getattr(forward_batch, "extend_seq_lens_cpu", None),
                    )
                swa_loc = swa_loc[:cache_rows]
        if get_bool_env_var(
            "SGLANG_DSV4_NPU_CP_VERIFY", "False"
        ) and _dsv4_npu_should_log_layer(layer_id, self._dsv4_num_hidden_layers):
            logger.warning(
                "DSV4 store_cache: layer_id=%s swa_k_shape=%s swa_k_norm=%.4f "
                "swa_loc_shape=%s swa_k_max=%.4f",
                layer_id,
                tuple(swa_k.shape),
                float(swa_k.float().norm().item()),
                tuple(swa_loc.shape),
                float(swa_k.float().abs().max().item()),
            )
        pool.set_swa_buffer(
            layer_id=layer_id,
            loc=swa_loc,
            cache=swa_k,
        )

    # PHASE-0 STUBS: all c4/c128 compressor / indexer paths are no-ops
    # while we surface the full forward chain. attention forward already
    # returns zeros for compress_ratio in (4, 128) (see forward()), so
    # whatever these compute would only feed a zero attention anyway.
    # The real impl of these (porting iforgetmyname's compressor/indexer
    # NPU kernels onto main's KV pool layout) is the bulk of the V4-NPU
    # attention port and lives behind these stubs.

    def forward_compress(self, *args, **kwargs):  # type: ignore[override]
        return None

    def forward_core_compressor(  # type: ignore[override]
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        layer_id: int,
        compressor,
    ) -> None:
        """Run the OUTER attention compressor on NPU.

        On CUDA, ``CompressorBackendMixin.forward_core_compressor`` calls
        ``compressor(x, forward_batch)`` (which produces compressed kv) and
        then writes the result via ``token_to_kv_pool.set_extra_key_buffer*``.
        On NPU, ``Compressor.forward_npu`` does the write inline (calls
        ``set_compress_buffer`` and ``set_compress_state_buffer`` itself), so
        we just trigger the compressor call and return — no separate set-
        buffer step. Gated by SGLANG_DSV4_NPU_REAL_COMPRESSOR; flag off keeps
        the previous stub (compressor never invoked, c4/c128 layers fall
        back to dense SWA in forward()).
        """
        if forward_batch.forward_mode.is_idle():
            return

        if not envs.SGLANG_DSV4_NPU_REAL_COMPRESSOR.get():
            return
        compressor(x, forward_batch)

    def forward_c4_indexer(  # type: ignore[override]
        self,
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: "ForwardBatch",
        c4_indexer=None,
        alt_streams=None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
    ) -> None:
        """Wire up ``forward_metadata.c4_topk_indices`` for c4 sparse attention.

        Stage 1 (this commit): seed ``c4_topk_indices`` with -1 sentinel so
        downstream ``_forward_compressed`` (when implemented for ratio=4) can
        read a well-shaped tensor. The real NPU compute path needs:
          1. q from ``c4_indexer.wq_b(q_lora)`` + rope + hadamard rotation
             (``compute_q`` in the model uses the tvm_ffi ``fused_rope``; on
             NPU we need to inline ``_v4_rope_inplace_npu`` + a torch hadamard)
          2. weights from ``c4_indexer.weights_proj(x)``
          3. indexer-K cache (currently absent — comes from the c4 indexer
             compressor write path which is also stubbed)
          4. ``torch_npu.npu_dynamic_quant`` for q quantization
          5. ``torch.ops.custom.npu_quant_lightning_indexer`` to produce the
             real top-k indices
        Each piece needs its own commit + 217 relaunch verification.
        """
        if forward_batch.forward_mode.is_idle():
            return
        # Stage 1 baseline: just seed c4_topk_indices=-1 sentinel for
        # _forward_compressed to read. Real path requires forking main's
        # Compressor / C4Indexer modules with NPU-style self-contained
        # impl (see ascend ref iforgetmyname/dsv4_release nsa_indexer.py
        # Compressor.forward_ori @ L241 — wkv + wgate + ape weighted sum
        # + norm + rope + write KV pool, all in-module, no backend
        # delegation). Stage 2A-I exploration (wq_b call + various store
        # patterns) showed the issue isn't the wq_b op itself but the
        # architectural mismatch — main's forward_compress is a triton
        # mixin path with no NPU equivalent; we must fork the model
        # modules, not the backend mixin.
        self.forward_metadata.c4_topk_indices = self._seed_c4_topk_indices(
            forward_batch
        )
