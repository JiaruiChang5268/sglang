from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    capture_routed_experts_if_allowed,
    select_experts,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


# Cache benchmark-only balanced routes per device/parallel configuration.  Keeping
# one shared cache avoids allocating the same fake route for every MoE layer.
_BALANCED_TOPK_CACHE: dict[tuple, torch.Tensor] = {}


def _get_balanced_topk_ids(
    num_tokens: int,
    topk: int,
    num_experts: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a deterministic, TP-staggered route with uniform expert counts."""
    if num_tokens == 0:
        return torch.empty((0, topk), dtype=dtype, device=device)
    if topk <= 0 or topk > num_experts:
        raise ValueError(
            f"Invalid balanced MoE routing shape: {topk=} and {num_experts=}"
        )

    parallel = get_parallel()
    tp_rank = parallel.tp_rank
    tp_size = parallel.tp_size
    cache_key = (device, dtype, topk, num_experts, tp_rank, tp_size)
    cached = _BALANCED_TOPK_CACHE.get(cache_key)

    if cached is None or cached.shape[0] < num_tokens:
        # Match the requested ordering: all even expert IDs, then all odd IDs,
        # rotated by one TP partition on each rank.  At least 16K token rows are
        # cached (equivalent to repeat(512) for 256 experts and top-k 8), and the
        # cache grows if a larger prefill arrives.
        expert_ids = torch.arange(num_experts, dtype=dtype, device=device)
        expert_ids = torch.cat((expert_ids[::2], expert_ids[1::2]))
        shift = tp_rank * num_experts // tp_size
        expert_ids = torch.cat((expert_ids[shift:], expert_ids[:shift]))

        capacity = max(num_tokens, 16 * 1024)
        num_values = capacity * topk
        repeats = (num_values + num_experts - 1) // num_experts
        cached = expert_ids.repeat(repeats)[:num_values].view(capacity, topk)
        _BALANCED_TOPK_CACHE[cache_key] = cached

    return cached[:num_tokens]


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
    force_balanced_topk: Optional[bool] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias
    use_npu_moe_gating_top_k = get_bool_env_var("USE_NPU_MOE_GATING_TOP_K")

    # sqrtsoftplus (DSV4 noaux_tc): top-k over (scores + bias); weights from
    # un-biased scores. The custom op fuses softplus/sqrt/topk/gather/norm/cast.
    if topk_config.scoring_func == "sqrtsoftplus":
        if use_npu_moe_gating_top_k:
            routed_scaling_factor = (
                topk_config.routed_scaling_factor
                if topk_config.apply_routed_scaling_factor_on_output
                else 1.0
            )
            topk_weights, topk_ids, _ = torch.ops.custom.npu_moe_gating_top_k(
                x=router_logits.to(torch.float32),
                k=topk_config.top_k,
                bias=(
                    correction_bias.to(torch.float32)
                    if correction_bias is not None
                    else None
                ),
                input_ids=None,
                tid2eid=None,
                routed_scaling_factor=float(routed_scaling_factor),
                norm_type=2,
            )
        else:
            scores = torch.nn.functional.softplus(router_logits.float()).sqrt()
            scores_for_choice = (
                scores + correction_bias.unsqueeze(0).float()
                if correction_bias is not None
                else scores
            )
            _, topk_ids = torch.topk(
                scores_for_choice, k=topk_config.top_k, dim=-1, sorted=False
            )
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = scores.gather(1, topk_ids)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            elif topk_config.routed_scaling_factor is not None:
                topk_weights = topk_weights * topk_config.routed_scaling_factor
        topk_weights = topk_weights.to(torch.float32)

    # Fast path: simple top-k without grouped routing and bias
    elif not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=topk_config.top_k,
        )

        if renormalize:
            topk_weights = l1_norm(
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
        topk_weights = topk_weights.to(torch.float32)

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            # num_expert_group and topk_group in some topk_config without group is None, (not supported by this ops)
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=0,
            # 1 for sigmoid, 0 for softmax
            norm_type=1,
            routed_scaling_factor=(
                1 if renormalize else topk_config.routed_scaling_factor
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # torch native is not yet supported num_token_non_padded
    # Fallback to torch native implementation
    else:
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            allow_round_robin_simulation=force_balanced_topk is not False,
        )

    simulate_uniform_experts = envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
    simulate_round_robin_experts = (
        envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )
    if simulate_uniform_experts and simulate_round_robin_experts:
        raise ValueError(
            "SGLANG_SIMULATE_UNIFORM_EXPERTS and "
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS are mutually exclusive"
        )

    if simulate_uniform_experts:
        # Preserve the existing cross-platform simulation semantics on NPU.
        num_tokens, topk = topk_ids.shape
        num_experts = router_logits.shape[1]
        offsets = torch.randint(
            0, num_experts, (num_tokens, 1), device=topk_ids.device
        )
        steps = torch.arange(topk, device=topk_ids.device).unsqueeze(0)
        step = max(num_experts // topk, 1)
        topk_ids = ((offsets + steps * step) % num_experts).to(topk_ids.dtype)
        topk_weights = torch.full_like(topk_weights, 1.0 / topk)
    elif simulate_round_robin_experts and force_balanced_topk is not False:
        # Benchmark-only: force deterministic, uniformly distributed expert IDs.
        # Keep the router-produced weights so this changes only expert placement.
        topk_ids = _get_balanced_topk_ids(
            hidden_states.shape[0],
            topk_ids.shape[1],
            router_logits.shape[1],
            device=topk_ids.device,
            dtype=topk_ids.dtype,
        )

    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
