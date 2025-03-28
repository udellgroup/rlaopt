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


def _get_D_matern(Ai_lazy, Aj_lazy):
    """Compute distance matrix for Matern kernels."""
    D = ((Ai_lazy - Aj_lazy) ** 2).sum(dim=2).sqrt()
    return D


def _kernel_computation_rbf(Ai_lazy, Aj_lazy, kernel_params):
    """Compute RBF kernel matrix."""
    D = ((Ai_lazy - Aj_lazy) ** 2).sum(dim=2)
    K_lazy = (-D / (2 * kernel_params["sigma"] ** 2)).exp()
    return K_lazy


def _kernel_computation_laplace(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Laplace kernel matrix."""
    D = (Ai_lazy - Aj_lazy).abs().sum(dim=2)
    K_lazy = (-D / kernel_params["sigma"]).exp()
    return K_lazy


def _kernel_computation_matern12(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Matern-1/2 kernel matrix."""
    D = _get_D_matern(Ai_lazy, Aj_lazy)
    K_lazy = (-D / kernel_params["sigma"]).exp()
    return K_lazy


def _kernel_computation_matern32(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Matern-3/2 kernel matrix."""
    D = _get_D_matern(Ai_lazy, Aj_lazy)
    D_adj = (3**0.5) * D / kernel_params["sigma"]
    K_lazy = (1 + D_adj) * (-D_adj).exp()
    return K_lazy


def _kernel_computation_matern52(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Matern-5/2 kernel matrix."""
    D = _get_D_matern(Ai_lazy, Aj_lazy)
    D_adj = (5**0.5) * D / kernel_params["sigma"]
    K_lazy = (1 + D_adj + 5 * D**2 / (3 * kernel_params["sigma"] ** 2)) * (
        -D_adj
    ).exp()
    return K_lazy


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
