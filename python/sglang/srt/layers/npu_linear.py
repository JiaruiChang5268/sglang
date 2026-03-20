"""NPU-specific linear layers with optimized kernels."""

import logging
from typing import Optional

import torch
from torch.nn import Parameter

try:
    import torch_npu

    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_npu_catcoc_shmem_addr
from sglang.srt.distributed.utils import split_tensor_along_last_dim
from sglang.srt.layers.linear import LinearBase, divide
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import set_weight_attrs
from sglang.srt.model_loader.weight_utils import pad_or_narrow_weight

logger = logging.getLogger(__name__)


class CatcocRowParallelLinear(LinearBase):
    """NPU-optimized Row Parallel Linear layer using catcoc_matmul_allreduce.

    This layer fuses matrix multiplication and allreduce operations using
    the catcoc_matmul_allreduce operator, providing better performance
    for distributed inference on NPU devices.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
        use_nz_format: bool = True,
        shmem_size: int = 1024 * 1024 * 1024,  # 1GB default
        team_id: int = 0,
    ):
        if not _NPU_AVAILABLE:
            raise RuntimeError(
                "NPU support not available. Please install torch_npu and shmem."
            )

        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.input_is_parallel = input_is_parallel
        self.use_nz_format = use_nz_format
        self.team_id = team_id

        # Tensor parallelism setup
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size

        # Shard the weight matrix along the input dimension
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.use_presharded_weights = use_presharded_weights

        # Shared memory setup - use global shmem instead of local initialization
        self.shmem_size = shmem_size
        self.shmem_addr = None  # Will be set from global state in forward

        # Create weight parameter
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                dtype=params_dtype or torch.float16,
            )
        )
        set_weight_attrs(
            self.weight,
            {
                "input_dim": 1,  # Fixed: input dimension is 1, not 0
                "output_dim": 0,  # Fixed: output dimension is 0, not 1
                "packed_dim": self.input_size,
                "pack_factor": 1,
                "weight_loader": self.weight_loader,
            },
        )

        # Create bias if needed
        if bias:
            self.bias = Parameter(torch.zeros(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """Load weight with proper sharding for row parallelism."""
        input_dim = getattr(param, "input_dim", None)

        param_data = param.data

        if (
            input_dim is not None
            and not self.use_presharded_weights
            and self.tp_size > 1
        ):
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size

            # Padding for special cases
            end_idx = start_idx + shard_size
            if end_idx > loaded_weight.shape[input_dim]:
                loaded_weight = pad_or_narrow_weight(
                    loaded_weight, input_dim, start_idx, shard_size
                )
            else:
                loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)

    def forward(self, input_: torch.Tensor):
        """Forward pass using catcoc_matmul_allreduce when available."""
        # Get global shmem address (only once per forward call)
        if self.shmem_addr is None:
            self.shmem_addr = get_npu_catcoc_shmem_addr()

        # Handle input sharding
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # Split input across TP ranks
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Get shapes for output allocation
        batch_size = (
            input_parallel.shape[0]
            if input_parallel.dim() == 2
            else input_parallel.shape[:-1].numel()
        )
        if input_parallel.dim() > 2:
            input_parallel = input_parallel.view(-1, input_parallel.shape[-1])

        output_shape = list(input_parallel.shape[:-1]) + [self.output_size]

        # Try to use catcoc operation
        if (
            self.shmem_addr is not None
            and self.tp_size > 1
            and hasattr(torch.ops, "npu")
            and hasattr(torch.ops.npu, "catcoc_matmul_allreduce")
        ):
            try:
                # Prepare output tensor
                output = torch.empty(
                    output_shape,
                    dtype=input_parallel.dtype,
                    device=input_parallel.device,
                ).contiguous()

                # Prepare weight for operation
                weight_t = self.weight.t().contiguous()

                if self.use_nz_format:
                    # Convert weight to NZ format
                    weight_nz = torch_npu.npu_format_cast(weight_t, 29)
                    torch.ops.npu.catcoc_matmul_allreduce(
                        input_parallel,
                        weight_nz,
                        output,
                        self.shmem_addr,
                        self.team_id,
                        format_mode="NZ",
                    )
                else:
                    torch.ops.npu.catcoc_matmul_allreduce(
                        input_parallel, weight_t, output, self.shmem_addr, self.team_id
                    )

                torch.npu.synchronize()

                # Handle bias
                if self.bias is not None and not self.skip_bias_add:
                    output = output + self.bias

                bias_out = self.bias if self.skip_bias_add else None
                return output, bias_out

            except Exception as e:
                logger.warning(
                    f"Catcoc operation failed: {e}, falling back to standard approach"
                )

        # Fallback to standard approach
        output = torch.matmul(input_parallel, self.weight.t())

        # Manual allreduce if needed
        if self.tp_size > 1:
            from sglang.srt.distributed import tensor_model_parallel_all_reduce

            output = tensor_model_parallel_all_reduce(output)

        # Handle bias
        if self.bias is not None and not self.skip_bias_add:
            output = output + self.bias

        bias_out = self.bias if self.skip_bias_add else None
        return output, bias_out
