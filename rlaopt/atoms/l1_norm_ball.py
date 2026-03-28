"""L1-norm ball constraint atom for optimization."""

from numbers import Real

import torch

from rlaopt.atoms.atom import Atom, AtomDecomposition
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


class L1NormBall(Atom):
    """L1-norm ball constraint enforcing ||x||_1 <= radius.

    This atom represents the indicator function of the L1-norm ball:
        0 if ||x||_1 <= radius, +inf otherwise.

    Args:
        x: Expression to constrain.
        radius: Non-negative radius of the L1-norm ball (default: 1.0).
    """

    def __init__(self, x: Expression, radius: float | int | torch.Tensor = 1.0):
        """Initialize the L1-norm ball constraint atom."""
        radius = _validate_radius(radius)
        super().__init__(exprs={"x": x}, buffers={"radius": radius})

    def is_smooth(self) -> bool:
        """L1-norm ball indicator is non-smooth."""
        return False

    def forward(self) -> torch.Tensor:
        """Evaluate the indicator function of the L1-norm ball."""
        value = self.get_input("x").forward()
        radius = self.get_buffer("radius").to(device=value.device, dtype=value.dtype)
        norm = torch.sum(torch.abs(value))
        satisfied = (norm <= radius).item()
        return _indicator(satisfied, value.device, value.dtype)

    def is_proxable(self) -> bool:
        """Check if the proximal operator is computable."""
        return isinstance(self.get_input("x"), Variable)

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Project onto the L1-norm ball (prox of indicator)."""
        radius = self.get_buffer("radius")

        def project_onto_l1_ball(x: torch.Tensor) -> torch.Tensor:
            radius_t = radius.to(device=x.device, dtype=x.dtype)
            if radius_t.item() <= 0:
                return torch.zeros_like(x)

            flat = x.reshape(-1)
            abs_flat = torch.abs(flat)
            if (torch.sum(abs_flat) <= radius_t).item():
                return x

            sorted_abs, _ = torch.sort(abs_flat, descending=True)
            cumsum = torch.cumsum(sorted_abs, dim=0)
            idx = torch.arange(
                1, sorted_abs.numel() + 1, device=flat.device, dtype=flat.dtype
            )
            cond = sorted_abs * idx > (cumsum - radius_t)
            rho = torch.nonzero(cond, as_tuple=False)[-1].item()
            theta = (cumsum[rho] - radius_t) / (rho + 1)
            projected = torch.sign(flat) * torch.nn.functional.relu(abs_flat - theta)
            return projected.reshape(x.shape)

        return relevant_variable_values.apply(project_onto_l1_ball)

    def decompose(self) -> list[AtomDecomposition] | None:
        """Decompose the constraint if the input is affine."""
        input_expr = self.get_input("x")
        if not input_expr.is_affine():
            return None

        new_var = Variable.like(input_expr)
        radius = self.get_buffer("radius")
        new_atom = type(self)(new_var, radius=radius)
        return [AtomDecomposition(atom=new_atom, affine_expr=input_expr)]


def _validate_radius(radius: float | int | torch.Tensor) -> float | torch.Tensor:
    """Validate and normalize the radius parameter."""
    if isinstance(radius, Real):
        if radius < 0:
            raise ValueError("radius must be non-negative")
        return float(radius)
    if torch.is_tensor(radius):
        if radius.numel() != 1:
            raise ValueError("radius must be a scalar tensor")
        if torch.any(radius < 0):
            raise ValueError("radius must be non-negative")
        return radius
    raise TypeError(
        f"radius must be float, int, or Tensor, got {type(radius).__name__}"
    )


def _indicator(
    satisfied: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return 0 if satisfied, infinity otherwise."""
    if satisfied:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return torch.tensor(torch.inf, device=device, dtype=dtype)
