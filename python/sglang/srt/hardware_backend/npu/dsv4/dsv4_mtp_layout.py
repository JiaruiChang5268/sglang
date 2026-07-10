"""Request-aware cache-location layout helpers for DSV4-NPU MTP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DSV4MTPStepCacheLocs:
    """Cache locations used by one top-k=1 MTP draft step.

    Full/SWA/state locations have one row per graph batch entry. Rows beyond
    ``real_batch_size`` are zero-filled dummy slots. Compressed locations are
    densely packed for real requests that cross a compression boundary.
    """

    out_full_loc: torch.Tensor
    out_swa_loc: torch.Tensor
    out_c4_loc: torch.Tensor
    out_c128_loc: torch.Tensor
    out_c4_state_loc: torch.Tensor
    out_c128_state_loc: torch.Tensor
    real_batch_size: int
    padded_batch_size: int


def _pad_with_dummy_rows(
    loc: torch.Tensor, real_batch_size: int, padded_batch_size: int
) -> torch.Tensor:
    if real_batch_size == padded_batch_size:
        return loc
    padded = loc.new_zeros((padded_batch_size,))
    padded[:real_batch_size].copy_(loc)
    return padded


def _first_mismatch(actual: torch.Tensor, expected: torch.Tensor) -> int | None:
    mismatch = torch.nonzero(actual != expected, as_tuple=False)
    return int(mismatch[0].item()) if mismatch.numel() > 0 else None


def build_dsv4_topk1_step_cache_locs(
    *,
    dense_full_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token_pool: Any,
    step_id: int,
    num_steps: int,
    padded_batch_size: int,
    validate: bool = False,
) -> DSV4MTPStepCacheLocs:
    """Build one DSV4 MTP step from request tables, not the reserve bundle.

    ``dense_full_locs`` has the EAGLE request-major layout
    ``[request, step]`` for top-k=1. The allocation bundle is intentionally not
    an input: it contains every newly reserved slot and may have a larger or
    even variable per-request stride when overlap-mode reservations are reused.

    Graph replay can pad ``padded_batch_size`` after ``dense_full_locs`` was
    built. The real batch size is therefore derived from the dense draft locs;
    padded rows use slot 0, the NPU dummy/sentinel slot.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not 0 <= step_id < num_steps:
        raise ValueError(f"step_id={step_id} is outside [0, {num_steps})")
    if dense_full_locs.ndim != 1:
        raise ValueError(
            f"dense_full_locs must be 1D, got shape={tuple(dense_full_locs.shape)}"
        )
    if dense_full_locs.numel() % num_steps != 0:
        raise ValueError(
            "dense MTP cache-loc length is not request-major: "
            f"numel={dense_full_locs.numel()}, num_steps={num_steps}"
        )

    real_batch_size = dense_full_locs.numel() // num_steps
    if padded_batch_size < real_batch_size:
        raise ValueError(
            f"padded_batch_size={padded_batch_size} is smaller than "
            f"real_batch_size={real_batch_size}"
        )
    if req_pool_indices.numel() < real_batch_size:
        raise ValueError(
            f"req_pool_indices has {req_pool_indices.numel()} rows, expected at "
            f"least {real_batch_size}"
        )
    if seq_lens.numel() < real_batch_size:
        raise ValueError(
            f"seq_lens has {seq_lens.numel()} rows, expected at least "
            f"{real_batch_size}"
        )

    req_indices = req_pool_indices[:real_batch_size].to(torch.int64)
    logical_positions = seq_lens[:real_batch_size].to(torch.int64) + step_id
    full_real = dense_full_locs.reshape(real_batch_size, num_steps)[:, step_id]

    if validate:
        expected_full = req_to_token_pool.req_to_token[
            req_indices, logical_positions
        ].to(full_real.dtype)
        mismatch_idx = _first_mismatch(full_real, expected_full)
        if mismatch_idx is not None:
            raise RuntimeError(
                "DSV4 NPU MTP full cache-loc invariant failed: "
                f"step={step_id}, batch_row={mismatch_idx}, "
                f"req_pool_idx={int(req_indices[mismatch_idx].item())}, "
                f"logical_position={int(logical_positions[mismatch_idx].item())}, "
                f"actual={int(full_real[mismatch_idx].item())}, "
                f"expected={int(expected_full[mismatch_idx].item())}"
            )

    def gather_raw_table(name: str) -> torch.Tensor:
        table = getattr(req_to_token_pool, name)
        return table[req_indices, logical_positions]

    def gather_compressed_table(name: str, ratio: int) -> torch.Tensor:
        table = getattr(req_to_token_pool, name)
        boundary = ((logical_positions + 1) % ratio) == 0
        return table[req_indices[boundary], logical_positions[boundary] // ratio]

    swa_real = gather_raw_table("req_to_token_swa")
    c4_state_real = gather_raw_table("req_to_token_c4_state")
    c128_state_real = gather_raw_table("req_to_token_c128_state")

    return DSV4MTPStepCacheLocs(
        out_full_loc=_pad_with_dummy_rows(
            full_real, real_batch_size, padded_batch_size
        ),
        out_swa_loc=_pad_with_dummy_rows(swa_real, real_batch_size, padded_batch_size),
        out_c4_loc=gather_compressed_table("req_to_token_c4", 4),
        out_c128_loc=gather_compressed_table("req_to_token_c128", 128),
        out_c4_state_loc=_pad_with_dummy_rows(
            c4_state_real, real_batch_size, padded_batch_size
        ),
        out_c128_state_loc=_pad_with_dummy_rows(
            c128_state_real, real_batch_size, padded_batch_size
        ),
        real_batch_size=real_batch_size,
        padded_batch_size=padded_batch_size,
    )
