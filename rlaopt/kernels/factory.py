from typing import Callable
import torch

from .base import _KernelLinOp, _DistributedKernelLinOp
from .configs import KernelConfig
from .utils import _row_oracle_matvec, _block_chunk_matvec
from rlaopt.linops import ScaleLinOp


def _create_kernel_classes(
    kernel_name: str, kernel_computation_fn: Callable
) -> tuple[type, type]:
    """Factory function to create kernel linear operator classes.

    Args:
        kernel_name: Name of the kernel (used for caching)
        kernel_computation_fn: Function that computes the kernel matrix

    Returns:
        A tuple of (KernelLinOp, DistributedKernelLinOp) classes
    """
    # Helper function to apply scaling to any kernel operator
    def apply_scaling(operator, kernel_config):
        # Get scaling factor (default to 1.0 if not specified)
        scaling = getattr(kernel_config, "const_scaling", 1.0)
        # Always return a ScaleLinOp, even if scaling is 1.0
        return ScaleLinOp(operator, scaling)

    # Define the kernel class that users will instantiate
    class KernelLinOp(_KernelLinOp):
        def __new__(
            cls, A1: torch.Tensor, A2: torch.Tensor, kernel_config: KernelConfig
        ):
            # Create the base kernel operator
            base_op = _KernelLinOp.__new__(cls)
            _KernelLinOp.__init__(
                base_op,
                A1=A1,
                A2=A2,
                kernel_config=kernel_config,
                _kernel_computation_fn=kernel_computation_fn,
            )

            # Apply scaling and return
            return apply_scaling(base_op, kernel_config)

    # Define the distributed kernel class
    class DistributedKernelLinOp(_DistributedKernelLinOp):
        def __new__(
            cls,
            A1: torch.Tensor,
            A2: torch.Tensor,
            kernel_config: KernelConfig,
            devices: set[torch.device],
            use_full_kernel: bool = True,
        ):
            # Create the base distributed kernel operator
            base_op = _DistributedKernelLinOp.__new__(cls)
            _DistributedKernelLinOp.__init__(
                base_op,
                A1=A1,
                A2=A2,
                kernel_config=kernel_config,
                devices=devices,
                use_full_kernel=use_full_kernel,
                _kernel_computation_fn=kernel_computation_fn,
                _row_oracle_matvec_fn=_row_oracle_matvec,
                _block_chunk_matvec_fn=_block_chunk_matvec,
                _cacheable_kernel_name=kernel_name,
            )

            # Apply scaling and return
            return apply_scaling(base_op, kernel_config)

    # Rename the classes with proper kernel name
    KernelLinOp.__name__ = f"{kernel_name}LinOp"
    KernelLinOp.__qualname__ = f"{kernel_name}LinOp"
    DistributedKernelLinOp.__name__ = f"Distributed{kernel_name}LinOp"
    DistributedKernelLinOp.__qualname__ = f"Distributed{kernel_name}LinOp"

    # Add docstrings
    KernelLinOp.__doc__ = (
        f"{kernel_name} kernel linear operator. "
        "Automatically applies scaling from kernel_config.const_scaling "
        "(defaults to 1.0 if not in kernel_config)."
    )

    DistributedKernelLinOp.__doc__ = (
        f"Distributed {kernel_name} kernel linear operator. "
        "Automatically applies scaling from kernel_config.const_scaling "
        "(defaults to 1.0 if not in kernel_config)."
    )

    return KernelLinOp, DistributedKernelLinOp
