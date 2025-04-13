from typing import Dict, Callable, Set, Tuple
import torch

from .base import _KernelLinOp, _DistributedKernelLinOp
from .utils import _check_kernel_params, _row_oracle_matvec


def _create_kernel_classes(
    kernel_name: str, kernel_computation_fn: Callable
) -> Tuple[_KernelLinOp, _DistributedKernelLinOp]:
    """Factory function to create kernel linear operator classes.

    Args:
        kernel_name: Name of the kernel (used for caching)
        kernel_computation_fn: Function that computes the kernel matrix

    Returns:
        A tuple of (_KernelLinOp, _DistributedKernelLinOp) classes
    """
    # Define the class initialization method that will be used
    def kernel_init(self, A: torch.Tensor, kernel_params: Dict):
        _KernelLinOp.__init__(
            self,
            A=A,
            kernel_params=kernel_params,
            _check_kernel_params_fn=_check_kernel_params,
            _kernel_computation_fn=kernel_computation_fn,
        )

    # Define the distributed class initialization method
    def distributed_kernel_init(
        self,
        A: torch.Tensor,
        kernel_params: Dict,
        devices: Set[torch.device],
        # compute_device: Optional[torch.device] = None,
    ):
        _DistributedKernelLinOp.__init__(
            self,
            A=A,
            kernel_params=kernel_params,
            devices=devices,
            # compute_device=compute_device,
            _check_kernel_params_fn=_check_kernel_params,
            _kernel_computation_fn=kernel_computation_fn,
            _row_oracle_matvec_fn=_row_oracle_matvec,
            _cacheable_kernel_name=kernel_name,
        )

    # Create class names with proper capitalization
    class_name = f"{kernel_name}LinOp"
    distributed_class_name = f"Distributed{class_name}"

    # Dynamically create classes with the specified names
    KernelLinOp = type(
        class_name,  # Name of the class
        (_KernelLinOp,),  # Base classes
        {"__init__": kernel_init},  # Class dictionary (attributes and methods)
    )

    DistributedKernelLinOp = type(
        distributed_class_name,
        (_DistributedKernelLinOp,),
        {"__init__": distributed_kernel_init},
    )

    # Add docstrings for better help() output
    KernelLinOp.__doc__ = f"{kernel_name} kernel linear operator."
    DistributedKernelLinOp.__doc__ = (
        f"Distributed {kernel_name} kernel linear operator with row and block oracles "
        "that share worker processes."
    )

    return KernelLinOp, DistributedKernelLinOp
