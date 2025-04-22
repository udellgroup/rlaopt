from pykeops.torch import LazyTensor

from .configs import KernelConfig
from .factory import _create_kernel_classes


__all__ = [
    "RBFLinOp",
    "DistributedRBFLinOp",
    "LaplaceLinOp",
    "DistributedLaplaceLinOp",
    "Matern12LinOp",
    "DistributedMatern12LinOp",
    "Matern32LinOp",
    "DistributedMatern32LinOp",
    "Matern52LinOp",
    "DistributedMatern52LinOp",
]


_RBF_NAME = "RBF"
_LAPLACE_NAME = "Laplace"
_MATERN12_NAME = "Matern12"
_MATERN32_NAME = "Matern32"
_MATERN52_NAME = "Matern52"

_SQRT3 = 3**0.5
_SQRT5 = 5**0.5


def _get_scaled_diff(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: KernelConfig
):
    """Compute scaled difference for kernels."""
    return (Ai_lazy - Aj_lazy) / kernel_config.lengthscale


def _get_scaled_diff_matern(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: KernelConfig
):
    """Compute scaled distance matrix for Matern kernels."""
    D = _get_scaled_diff(Ai_lazy, Aj_lazy, kernel_config)
    return (D**2).sum(dim=2).sqrt()


def _apply_const_scaling(K_lazy: LazyTensor, kernel_config: KernelConfig):
    """Apply constant scaling to the kernel matrix."""
    return kernel_config.const_scaling * K_lazy


def _kernel_computation_rbf(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: KernelConfig
):
    """Compute RBF kernel."""
    D = _get_scaled_diff(Ai_lazy, Aj_lazy, kernel_config)
    D = (D**2).sum(dim=2)
    return _apply_const_scaling((-D / 2).exp(), kernel_config)


def _kernel_computation_laplace(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: KernelConfig
):
    """Compute Laplace kernel."""
    D = _get_scaled_diff(Ai_lazy, Aj_lazy, kernel_config)
    D = D.abs().sum(dim=2)
    return _apply_const_scaling((-D).exp(), kernel_config)


def _kernel_computation_matern12(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: KernelConfig
):
    """Compute Matern-1/2 kernel."""
    D = _get_scaled_diff_matern(Ai_lazy, Aj_lazy, kernel_config)
    return _apply_const_scaling((-D).exp(), kernel_config)


def _kernel_computation_matern32(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: KernelConfig
):
    """Compute Matern-3/2 kernel."""
    D = _get_scaled_diff_matern(Ai_lazy, Aj_lazy, kernel_config)
    return _apply_const_scaling((1 + _SQRT3 * D) * (-_SQRT3 * D).exp(), kernel_config)


def _kernel_computation_matern52(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: KernelConfig
):
    """Compute Matern-5/2 kernel."""
    D = _get_scaled_diff_matern(Ai_lazy, Aj_lazy, kernel_config)
    return _apply_const_scaling(
        (1 + _SQRT5 * D + 5 / 3 * D**2) * (-_SQRT5 * D).exp(), kernel_config
    )


RBFLinOp, DistributedRBFLinOp = _create_kernel_classes(
    kernel_name=_RBF_NAME,
    kernel_computation_fn=_kernel_computation_rbf,
)

LaplaceLinOp, DistributedLaplaceLinOp = _create_kernel_classes(
    kernel_name=_LAPLACE_NAME,
    kernel_computation_fn=_kernel_computation_laplace,
)

Matern12LinOp, DistributedMatern12LinOp = _create_kernel_classes(
    kernel_name=_MATERN12_NAME,
    kernel_computation_fn=_kernel_computation_matern12,
)

Matern32LinOp, DistributedMatern32LinOp = _create_kernel_classes(
    kernel_name=_MATERN32_NAME,
    kernel_computation_fn=_kernel_computation_matern32,
)

Matern52LinOp, DistributedMatern52LinOp = _create_kernel_classes(
    kernel_name=_MATERN52_NAME,
    kernel_computation_fn=_kernel_computation_matern52,
)
