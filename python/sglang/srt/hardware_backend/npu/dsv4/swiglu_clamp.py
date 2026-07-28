# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _swiglu_clamp_kernel(
    x_ptr,
    out_ptr,
    total_rows,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    LIMIT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_per_core = (total_rows - 1) // NUM_CORES + 1
    row_begin = pid * rows_per_core
    if row_begin >= total_rows:
        return

    row_end = tl.minimum(row_begin + rows_per_core, total_rows)
    cols = tl.arange(0, HALF_COLS)

    for row_idx in range(row_begin, row_end):
        input_base = row_idx * TOTAL_COLS
        gate = tl.load(x_ptr + input_base + cols)
        up = tl.load(x_ptr + input_base + HALF_COLS + cols)

        # DeepSeek V4 clamps in the input dtype before evaluating SiLU.
        gate = tl.minimum(gate, LIMIT).to(x_ptr.dtype.element_ty)
        up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT).to(x_ptr.dtype.element_ty)

        gate_fp32 = gate.to(tl.float32)
        up_fp32 = up.to(tl.float32)
        out = gate_fp32 * tl.sigmoid(gate_fp32) * up_fp32

        output_base = row_idx * HALF_COLS
        tl.store(
            out_ptr + output_base + cols,
            out.to(out_ptr.dtype.element_ty),
        )


def npu_swiglu_clamp(x: torch.Tensor, limit: float) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, but got shape={tuple(x.shape)}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"The last dimension must be even, but got {x.shape[-1]}")
    if not x.is_contiguous():
        raise ValueError("npu_swiglu_clamp requires contiguous input")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Only BF16/FP16 are supported, but got dtype={x.dtype}")

    rows, total_cols = x.shape
    half_cols = total_cols // 2
    out = torch.empty((rows, half_cols), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out

    _, num_vector_cores = get_device_properties()
    _swiglu_clamp_kernel[(num_vector_cores,)](
        x,
        out,
        rows,
        TOTAL_COLS=total_cols,
        HALF_COLS=half_cols,
        NUM_CORES=num_vector_cores,
        LIMIT=float(limit),
        multibuffer=True,
    )
    return out
