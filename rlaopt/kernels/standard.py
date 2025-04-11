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


def _get_scaled_diff(Ai_lazy, Aj_lazy, kernel_params):
    """Compute scaled difference for kernels."""
    return (Ai_lazy - Aj_lazy) / kernel_params["lengthscale"]


def _get_scaled_diff_matern(Ai_lazy, Aj_lazy, kernel_params):
    """Compute scaled distance matrix for Matern kernels."""
    D = _get_scaled_diff(Ai_lazy, Aj_lazy, kernel_params)
    return (D**2).sum(dim=2).sqrt()


def _kernel_computation_rbf(Ai_lazy, Aj_lazy, kernel_params):
    """Compute RBF kernel."""
    D = _get_scaled_diff(Ai_lazy, Aj_lazy, kernel_params)
    D = (D**2).sum(dim=2)
    return (-D / 2).exp()


def _kernel_computation_laplace(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Laplace kernel."""
    D = _get_scaled_diff(Ai_lazy, Aj_lazy, kernel_params)
    D = D.abs().sum(dim=2)
    return (-D).exp()


def _kernel_computation_matern12(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Matern-1/2 kernel."""
    D = _get_scaled_diff_matern(Ai_lazy, Aj_lazy, kernel_params)
    return (-D).exp()


def _kernel_computation_matern32(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Matern-3/2 kernel."""
    D = _get_scaled_diff_matern(Ai_lazy, Aj_lazy, kernel_params)
    D_adj = (3**0.5) * D
    return (1 + D_adj) * (-D_adj).exp()


def _kernel_computation_matern52(Ai_lazy, Aj_lazy, kernel_params):
    """Compute Matern-5/2 kernel."""
    D = _get_scaled_diff_matern(Ai_lazy, Aj_lazy, kernel_params)
    D_adj = (5**0.5) * D
    return (1 + D_adj + (D_adj**2) / 3) * (-D_adj).exp()


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
