"""Isotonic (monotonic non-decreasing) constraint atom for optimization."""

from math import isclose
from typing import Any
from warnings import warn

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom, AtomDecomposition
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


class Isotonic(Atom):
    """Isotonic constraint enforcing ``x[0] <= x[1] <= ... <= x[n-1]``.

    Args:
        x: 1-D Expression to constrain to be non-decreasing.
    """

    def __init__(self, x: Expression):
        """Initialize the isotonic constraint atom.

        Args:
            x: 1-D Expression to constrain to be non-decreasing.
        """
        super().__init__(exprs={"x": x}, buffers={})

    def is_smooth(self) -> bool:
        """Isotonic constraint is non-smooth."""
        return False

    def is_proxable(self) -> bool:
        """Check if the proximal operator is computable."""
        return isinstance(self.get_input("x"), Variable)

    def forward(self) -> torch.Tensor:
        """Evaluate the indicator of the isotonic constraint."""
        value = self.get_input("x").forward()
        satisfied = bool(torch.all(torch.diff(value) >= 0))
        return _indicator(satisfied, value.device, value.dtype)

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Compute the proximal operator of the isotonic constraint.

        The projection is the prox of an indicator function, so it is independent of
        ``prox_scaling``.
        """
        return relevant_variable_values.apply(_prox_isotonic.apply)

    def decompose(self) -> list[AtomDecomposition] | None:
        """Decompose the constraint if the input is affine."""
        input_expr = self.get_input("x")
        if not input_expr.is_affine():
            return None

        new_var = Variable.like(input_expr)
        new_atom = type(self)(new_var)
        return [AtomDecomposition(atom=new_atom, affine_expr=input_expr)]

    def _scale(self, scaling: float) -> Self:
        """Scale the constraint (scaling preserves the constraint set)."""
        if isclose(scaling, 0.0):
            warn(
                f"Scaling a {self.__class__.__name__} constraint by zero has no effect."
            )
        return self


class _prox_isotonic(torch.autograd.Function):
    """Projection onto the isotonic cone via PAVA, with a custom backward pass.

    The forward pass runs the O(n) stack-based Pool Adjacent Violators Algorithm to
    compute the projection.

    The projection is piecewise-linear: on each block the output is the block mean,
    so its dependence on the input is local to that block. The backward pass therefore
    reduces to a block-average of the incoming gradient over the same partition.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` onto the isotonic cone ``{x[0] <= ... <= x[n-1]}``.

        Args:
            ctx: Context object for saving information needed in the backward pass.
            x: 1-D input tensor.

        Returns:
            The Euclidean projection of ``x`` onto the isotonic cone.
        """
        # A plain Python list of floats (no device),
        # so the sequential PAVA loop avoids per-element device->host syncs.
        vals = x.detach().tolist()

        # Stack-based O(n) PAVA: each element is pushed once and total pops <= n.
        block_sum: list[float] = []
        block_cnt: list[int] = []
        for v in vals:
            s, c = v, 1
            while block_sum and block_sum[-1] / block_cnt[-1] > s / c:
                s += block_sum.pop()
                c += block_cnt.pop()
            block_sum.append(s)
            block_cnt.append(c)

        # Realize the projection on x's device/dtype as a per-block mean of x.
        num_blocks = len(block_cnt)
        block_id = torch.repeat_interleave(
            torch.arange(num_blocks, device=x.device),
            torch.tensor(block_cnt, device=x.device),
        )
        sums = x.new_zeros(num_blocks).index_add_(0, block_id, x)
        counts = x.new_zeros(num_blocks).index_add_(0, block_id, torch.ones_like(x))
        y = (sums / counts)[block_id]

        ctx.save_for_backward(block_id)
        ctx.num_blocks = num_blocks
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Block-average the incoming gradient over the saved PAVA partition.

        Args:
            ctx: Context object containing the saved block partition.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input.
        """
        (block_id,) = ctx.saved_tensors
        num_blocks = ctx.num_blocks
        gsum = grad_output.new_zeros(num_blocks).index_add_(0, block_id, grad_output)
        gcnt = grad_output.new_zeros(num_blocks).index_add_(
            0, block_id, torch.ones_like(grad_output)
        )
        return (gsum / gcnt)[block_id]


def _indicator(
    satisfied: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return 0 if satisfied, infinity otherwise."""
    if satisfied:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return torch.tensor(torch.inf, device=device, dtype=dtype)
