"""Euclidean projections onto Lp-norm balls."""

import functools

import torch


def _zero_for_nonpositive_radius(projection):
    """Normalize ``radius`` to a tensor and short-circuit a zero projection.

    Wraps a projection so it converts ``radius`` to a tensor on ``x``'s device
    and dtype and, when the radius is non-positive, returns the zero vector
    directly -- the projection onto the degenerate ball ``{0}``. This skips the
    main computation and also sidesteps the empty active set that the L1
    projection would otherwise hit at radius 0.
    """

    @functools.wraps(projection)
    def wrapper(x: torch.Tensor, radius: float | torch.Tensor) -> torch.Tensor:
        radius_t = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
        if radius_t.item() <= 0:
            return torch.zeros_like(x)
        return projection(x, radius_t)

    return wrapper


@_zero_for_nonpositive_radius
def project_onto_l1_ball(x: torch.Tensor, radius: float | torch.Tensor) -> torch.Tensor:
    """Project a 1D tensor onto the L1-norm ball ``{z : ||z||_1 <= radius}``.

    Args:
        x: Tensor to project.
        radius: Non-negative radius of the L1-norm ball.

    Returns:
        The Euclidean projection of ``x`` onto the L1-norm ball.
    """
    abs_x = torch.abs(x)
    if (torch.sum(abs_x) <= radius).item():
        return x
    sorted_abs = torch.sort(abs_x, descending=True).values
    cumsum = torch.cumsum(sorted_abs, dim=0)
    idx = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    cond = sorted_abs * idx > (cumsum - radius)
    rho = torch.nonzero(cond, as_tuple=False)[-1].item()
    theta = (cumsum[rho] - radius) / (rho + 1)
    return torch.sign(x) * torch.relu(abs_x - theta)


@_zero_for_nonpositive_radius
def project_onto_l2_ball(x: torch.Tensor, radius: float | torch.Tensor) -> torch.Tensor:
    """Project a 1D tensor onto the L2-norm ball ``{z : ||z||_2 <= radius}``.

    Args:
        x: Tensor to project.
        radius: Non-negative radius of the L2-norm ball.

    Returns:
        The Euclidean projection of ``x`` onto the L2-norm ball.
    """
    norm = torch.linalg.norm(x)
    if (norm <= radius).item():
        return x
    return (radius / norm) * x


@_zero_for_nonpositive_radius
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
    return torch.clamp(x, min=-radius, max=radius)
