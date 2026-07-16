from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _apply_dsv4_npu_swa_prefill_chunk_size(server_args: ServerArgs) -> None:
    """Apply the opt-in SWA-bounded prefill fallback before DP normalization.

    This stays in the imperative hook because ``chunked_prefill_size`` is not
    a model-overridable field. ``_handle_data_parallelism`` later divides the
    value by ``dp_size``, so the environment value is the effective per-DP
    chunk size.
    """
    if server_args.device != "npu":
        return

    chunk_size = envs.SGLANG_DSV4_NPU_SWA_PREFILL_CHUNK_SIZE.get()
    if chunk_size < 0:
        raise ValueError("SGLANG_DSV4_NPU_SWA_PREFILL_CHUNK_SIZE must be >= 0")
    if chunk_size == 0:
        return

    page_size = server_args.page_size
    if chunk_size % page_size != 0:
        raise ValueError(
            "SGLANG_DSV4_NPU_SWA_PREFILL_CHUNK_SIZE must be a multiple "
            f"of the DSV4 NPU page size ({page_size}), got {chunk_size}"
        )

    configured_chunk_size = server_args.chunked_prefill_size
    if configured_chunk_size not in (None, -1):
        if configured_chunk_size != chunk_size:
            logger.warning(
                "Ignoring SGLANG_DSV4_NPU_SWA_PREFILL_CHUNK_SIZE=%d because "
                "--chunked-prefill-size is already %d.",
                chunk_size,
                configured_chunk_size,
            )
        return

    uses_dp_attention = server_args.enable_dp_attention or server_args.enable_prefill_cp
    declared_chunk_size = chunk_size * (server_args.dp_size if uses_dp_attention else 1)
    server_args.chunked_prefill_size = declared_chunk_size
    logger.warning(
        "Experimental DSV4 NPU SWA-bounded prefill is enabled: "
        "effective per-DP chunk=%d, declared chunked_prefill_size=%d "
        "(previous value: %s).",
        chunk_size,
        declared_chunk_size,
        configured_chunk_size,
    )


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Residual imperative arm of the DeepSeek V4 defaults.

    The attention/page/window/MoE-runner declarations moved to the override
    registry (arg_groups/overrides.py: _deepseek_v4_overrides) and the
    kv-cache dtype default to the resolution pipeline
    (_deepseek_v4_kv_cache_dtype, invoked below at its legacy slot). This
    keeps, at the legacy slot: the ROCm env fill (env-write policy), the NPU
    SWA-bounded prefill opt-in, the max_running_requests fill (the speculative
    hook is a later writer of that field) and the validations.
    """
    from sglang.srt.utils import is_hip

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there; an explicit env var still overrides this.
    if is_hip() and not envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set():
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.
    from sglang.srt.arg_groups.overrides import (
        _deepseek_v4_kv_cache_dtype,
        run_post_process_pass,
    )

    run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)

    _apply_dsv4_npu_swa_prefill_chunk_size(server_args)

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 256
        logger.warning(
            f"Setting max_running_requests to {server_args.max_running_requests} for {model_arch}."
        )

    if server_args.speculative_algorithm is not None:
        assert (
            server_args.speculative_algorithm == "EAGLE"
        ), f"Only EAGLE speculative algorithm is supported for {model_arch}"
        assert (
            server_args.speculative_eagle_topk == 1
        ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    if not server_args.enable_prefill_cp:
        return

    if server_args.cp_strategy not in ("interleave", "zigzag"):
        raise ValueError(
            "DeepSeekV4 only supports interleave/zigzag CP strategy, "
            f"got {server_args.cp_strategy}"
        )

    # DeepSeek V4 always drives CP through the DSA (NSA-family) runtime path,
    # never the MLA one. The first _handle_legacy_cp_arguments() pass runs before
    # this model hook resolves attention_backend to "dsv4", so the canonical
    # --enable-prefill-cp may have been mirrored onto enable_prefill_context_parallel
    # (the MLA alias). Clear it here so the two legacy aliases are not both set --
    # _handle_context_parallelism() rejects that as mutually exclusive.
    server_args.enable_dsa_prefill_context_parallel = True
    server_args.enable_prefill_context_parallel = False
    server_args.dsa_prefill_cp_mode = (
        "round-robin-split"
        if server_args.cp_strategy == "interleave"
        else "in-seq-split"
    )
    server_args.enable_dp_attention = True
    server_args.moe_dense_tp_size = 1
    server_args.attn_cp_size = server_args.tp_size // server_args.dp_size
    if server_args.cp_strategy == "interleave":
        assert (
            server_args.dp_size == 1
        ), "For round-robin split mode, dp attention is not supported."
    assert (
        server_args.tp_size <= 8
    ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
    logger.warning(
        "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL because DeepSeekV4 "
        "context parallelism is enabled."
    )
    envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"strategy={server_args.cp_strategy}, "
        f"dp_size={server_args.dp_size}, moe_dense_tp_size={server_args.moe_dense_tp_size}, "
        f"attn_cp_size={server_args.attn_cp_size}, ep_size={server_args.ep_size}, tp_size={server_args.tp_size}"
    )
