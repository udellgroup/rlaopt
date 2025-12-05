"""Base _OperatorSplit class for representing composite objective functions.

In optimization, a composite function is a function that consists of a smooth term and a
(typically non-smooth) proximal term.

The core functionality enables efficient splitting methods by providing access
to the smooth component, its gradient, and the proximal operator for the
non-smooth component. This is useful in first-order optimization algorithms such
as proximal gradient descent and ADMM.
"""

from abc import ABC, abstractmethod

import torch

from rlaopt.atoms import Atom
from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.linops import _HVPLinOp


class _OperatorSplit(ABC):
    """Base class for operator splitting methods.

    Provides common functionality for splitting composite objectives
    into smooth (f) and proximal (r) components.

    Subclasses should perform their own validation and splitting in __init__,
    then call super().__init__(f, r) with the split components.
    """

    def __init__(self, f: Expression, r: list[Atom]):
        """Initialize the operator split with pre-split components.

        Args:
            f: The smooth part of the objective
            r: The proximal part as a list of atoms
        """
        self._f = f
        self._r = r

    @property
    def f(self) -> Expression:
        """Returns the smooth component of the composite function."""
        return self._f

    @property
    def r(self) -> list[Atom]:
        """Returns the proximal component of the composite function."""
        return self._r

    @property
    @abstractmethod
    def variable_values(self) -> TensorDict:
        """Returns the variable values associated with the composite function."""
        pass

    def func_f(self, variable_values: TensorDict) -> torch.Tensor:
        """Evaluate the smooth part of the objective function at the given variables.

        Args:
            variable_values: A dictionary of variables.

        Returns:
            The scalar value of the smooth part at variable_values.
        """
        return self._f.evaluate(variable_values)

    def grad_f(self, variable_values: TensorDict) -> TensorDict:
        """Compute the gradient of the smooth part of the objective function.

        Args:
            variable_values: A dictionary of variables.

        Returns:
            A dictionary of gradients with the same structure as variable_values.
        """
        return torch.func.grad(self.func_f)(variable_values)

    def hvp_f(self, variable_values: TensorDict, v: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian-vector product of the smooth part of the objective.

        Args:
            variable_values: A dictionary of variables.
            v: A torch tensor of shape (variable_values.flat_dim(),)
                representing the vector to multiply with the Hessian.

        Returns:
            The Hessian-vector product of the Hessian at variable_values and v.
        """
        hvp_op = _HVPLinOp(self._f, variable_values)
        return hvp_op @ v

    def prox(self, variable_values: TensorDict, eta: float) -> TensorDict:
        """Apply the proximal operator of r with step size eta to the variables.

        Args:
            variable_values: A dictionary of variables.
            eta: Step size or scaling factor for the proximal operator.

        Returns:
            Updated variables after applying the proximal operator.
        """
        for r in self._r:
            variable_values_update = r.prox(variable_values, eta)
            variable_values.update(variable_values_update)
        return variable_values
