"""Euclidean projections onto Lp-norm balls."""

import torch


def project_onto_l1_ball(x: torch.Tensor, radius: float | torch.Tensor) -> torch.Tensor:
    """Project a 1D tensor onto the L1-norm ball ``{z : ||z||_1 <= radius}``.

    Args:
        x: Tensor to project.
        radius: Non-negative radius of the L1-norm ball.

    Returns:
        The Euclidean projection of ``x`` onto the L1-norm ball.
    """
    radius_t = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
    if radius_t.item() <= 0:
        return torch.zeros_like(x)
    abs_x = torch.abs(x)
    if (torch.sum(abs_x) <= radius_t).item():
        return x
    sorted_abs = torch.sort(abs_x, descending=True).values
    cumsum = torch.cumsum(sorted_abs, dim=0)
    idx = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    cond = sorted_abs * idx > (cumsum - radius_t)
    rho = torch.nonzero(cond, as_tuple=False)[-1].item()
    theta = (cumsum[rho] - radius_t) / (rho + 1)
    return torch.sign(x) * torch.relu(abs_x - theta)


def project_onto_l2_ball(x: torch.Tensor, radius: float | torch.Tensor) -> torch.Tensor:
    """Project a 1D tensor onto the L2-norm ball ``{z : ||z||_2 <= radius}``.

    Args:
        x: Tensor to project.
        radius: Non-negative radius of the L2-norm ball.

    Returns:
        The Euclidean projection of ``x`` onto the L2-norm ball.
    """
    radius_t = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
    if radius_t.item() <= 0:
        return torch.zeros_like(x)
    norm = torch.linalg.norm(x)
    if (norm <= radius_t).item():
        return x
    return (radius_t / norm) * x


def project_onto_linf_ball(
    x: torch.Tensor, radius: float | torch.Tensor
) -> torch.Tensor:
    """Project a 1D tensor onto the L-infinity ball ``{z : ||z||_inf <= radius}``.

    Args:
        x: Tensor to project.
        radius: Non-negative radius of the L-infinity-norm ball.

    Returns:
        The Euclidean projection of ``x`` onto the L-infinity-norm ball, i.e.
        the elementwise clamp to ``[-radius, radius]``.
    """
    radius_t = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
    if radius_t.item() <= 0:
        return torch.zeros_like(x)
    bound = radius_t.item()
    return torch.clamp(x, min=-bound, max=bound)
