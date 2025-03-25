from rlaopt.kernels.base import (
    _KernelLinOp,
    _DistributedKernelLinOp,
)
from rlaopt.kernels.utils import _check_kernel_params, _row_oracle_matvec

__all__ = ["RBFLinOp", "DistributedRBFLinOp"]


_CACHEABLE_KERNEL_NAME = "rbf"


def _kernel_computation(Ai_lazy, Aj_lazy, kernel_params):
    """Compute RBF kernel matrix."""
    D = ((Ai_lazy - Aj_lazy) ** 2).sum(dim=2)
    K_lazy = (-D / (2 * kernel_params["sigma"] ** 2)).exp()
    return K_lazy


class RBFLinOp(_KernelLinOp):
    def __init__(self, A, kernel_params):
        super().__init__(
            A=A,
            kernel_params=kernel_params,
            _check_kernel_params_fn=_check_kernel_params,
            _kernel_computation_fn=_kernel_computation,
        )


class DistributedRBFLinOp(_DistributedKernelLinOp):
    """Distributed RBF linear operator with row and block oracles that share worker
    processes."""

    def __init__(
        self,
        A,
        kernel_params,
        devices,
        compute_device=None,
    ):
        """Initialize the distributed RBF linear operator."""
        super().__init__(
            A=A,
            kernel_params=kernel_params,
            devices=devices,
            compute_device=compute_device,
            _check_kernel_params_fn=_check_kernel_params,
            _kernel_computation_fn=_kernel_computation,
            _row_oracle_matvec_fn=_row_oracle_matvec,
            _cacheable_kernel_name=_CACHEABLE_KERNEL_NAME,
        )
