"""Polyhedron constraint atom for optimization."""

from functools import partial
from math import isclose
from typing import Callable
from warnings import warn

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


class Polyhedron(Atom):
    """Polyhedral constraint atom for linear equality and inequality constraints.

    A polyhedron is defined by:
        - Equality constraints: A @ x = b
        - Inequality constraints: lower <= C @ x <= upper

    The atom evaluates to 0 if all constraints are satisfied, and infinity
    otherwise (indicator function of the polyhedral set).

    Args:
        x: Variable to constrain.
        A: Equality constraint matrix (optional). If provided, b must also
            be provided.
        b: Equality constraint vector (optional). Required if A is provided.
        C: Inequality constraint matrix (optional). If None, uses identity
            (box constraints).
        lower: Lower bound vector for inequalities (optional).
        upper: Upper bound vector for inequalities (optional).

    Raises:
        ValueError: If A is provided but b is None.
        ValueError: If constraint dimensions are inconsistent.
        ValueError: If no constraints are provided (trivial polyhedron).
    """

    def __init__(
        self,
        x: Variable,
        A: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        lower: torch.Tensor | int | float | None = None,
        upper: torch.Tensor | int | float | None = None,
    ):
        """Initialize the polyhedral constraint atom.

        Args:
            x: Variable to constrain.
            A: Equality constraint matrix (optional).
            b: Equality constraint vector (optional).
            C: Inequality constraint matrix (optional).
            lower: Lower bound vector for inequalities (optional).
            upper: Upper bound vector for inequalities (optional).

        Raises:
            ValueError: If A is provided but b is None.
            ValueError: If constraint dimensions are inconsistent.
            ValueError: If no constraints are provided.
        """
        if (A is not None) and (b is None):
            raise ValueError("b cannot be None when A is not None")
        elif (A is None) and (b is not None):
            raise ValueError("A cannot be None when b is not None")

        ## Convert float/int bounds to tensors FIRST (before using .device/.dtype)
        if isinstance(lower, (int, float)):
            lower = torch.tensor(float(lower))
        if isinstance(upper, (int, float)):
            upper = torch.tensor(float(upper))

        # Validate input dimensional consistency
        _validate(A, C, b, lower, upper)

        # if upper is provided but not lower, set lower to -infinity
        if (upper is not None) and (lower is None):
            lower = torch.tensor(-torch.inf, device=upper.device, dtype=upper.dtype)
        # if lower is provided but not upper, set upper to infinity
        elif (lower is not None) and (upper is None):
            upper = torch.tensor(torch.inf, device=lower.device, dtype=lower.dtype)

        super().__init__(
            exprs={"x": x},
            buffers={"A": A, "b": b, "C": C, "lower": lower, "upper": upper},
            variable_names=["x"],
        )

        self._eval = _build_eval(
            self.get_buffer("A"),
            self.get_buffer("C"),
            self.get_buffer("b"),
            self.get_buffer("lower"),
            self.get_buffer("upper"),
        )

    def forward(self) -> torch.Tensor:
        """Evaluate the polyhedral constraint at the current variable value.

        Returns:
            torch.Tensor: 0.0 if constraints are satisfied, infinity otherwise.
        """
        value = self.get_input("x").forward()
        return self._eval(value)

    def is_smooth(self) -> bool:
        """Check if the polyhedral constraint is smooth.

        Returns:
            bool: Always False, as indicator functions are non-smooth.
        """
        return False

    def is_proxable(self) -> bool:
        """Check if the polyhedral constraint has a computable proximal operator.

        Returns:
            bool: Always False (general polyhedral projection not implemented).
        """
        return False

    def is_subsamplable(self) -> bool:
        """Check if the polyhedral constraint supports subsampling.

        Returns:
            bool: Always False, as constraints cannot be subsampled.
        """
        return False

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Polyhedral constraint does not have a prox operator in general."""
        raise NotImplementedError("Polyhedron is not proxable")

    def subsample(self, indices: torch.Tensor) -> Self:
        """Subsample the polyhedral constraint (not supported).

        Args:
            indices: Indices to subsample (unused).

        Returns:
            Polyhedron: Not applicable.

        Raises:
            NotImplementedError: Polyhedron constraints cannot be subsampled.
        """
        raise NotImplementedError("Polyhedron is not subsamplable")

    def _scale(self, scaling: float) -> Self:
        """Scale the polyhedral constraint."""
        if isclose(scaling, 0.0):
            warn(
                f"Scaling a {self.__class__.__name__} constraint by zero has no effect.",  # noqa: E501
            )
        return self  # Scaling does not change the constraint set


def _validate(A, C, b, lower, upper):
    """Validate dimensional consistency of constraint matrices and vectors.

    Args:
        A: Equality constraint matrix (optional).
        C: Inequality constraint matrix (optional).
        b: Equality constraint vector (optional).
        lower: Lower bound vector (optional).
        upper: Upper bound vector (optional).

    Raises:
        ValueError: If dimensions are inconsistent.
    """
    if A is not None and b is not None:
        _validate_equality_constraints(A, b)

    if C is not None:
        _validate_inequality_constraints(C, lower, upper)


def _validate_equality_constraints(A, b):
    """Validate equality constraint dimensions (A @ x = b)."""
    if A.dim() == 0:
        raise ValueError("A must be at least 1-dimensional")

    if A.dim() == 1:
        _validate_equality_vector(A, b)
    else:
        _validate_equality_matrix(A, b)


def _validate_equality_vector(A, b):
    """Validate hyperplane constraint (a^T x = b)."""
    if b.dim() != 0:
        raise ValueError("For 1D A (hyperplane), b must be a scalar")
    if _is_zero(A):
        raise ValueError("To define a valid constraint, a must be non-zero.")


def _validate_equality_matrix(A, b):
    """Validate matrix equality constraints (A @ x = b)."""
    if A.shape[0] > A.shape[1]:
        raise ValueError("Valid A must have more columns than rows!")
    if b.dim() != 1:
        raise ValueError("For 2D A, b must be 1D")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have matching row counts")
    if _is_zero(A):
        raise ValueError("To define a valid constraint, A must be non-zero")


def _validate_inequality_constraints(C, lower, upper):
    """Validate inequality constraint dimensions (lower <= C @ x <= upper)."""
    if C.dim() == 0:
        raise ValueError("C must be at least 1-dimensional")

    if C.dim() == 1:
        _validate_inequality_vector(C, lower, upper)
    else:
        _validate_inequality_matrix(C, lower, upper)


def _validate_inequality_vector(C, lower, upper):
    """Validate halfspace constraint (lower <= c^T x <= upper)."""
    if lower is not None and lower.dim() != 0:
        raise ValueError("For 1D C (halfspace), lower must be a scalar")
    if upper is not None and upper.dim() != 0:
        raise ValueError("For 1D C (halfspace), upper must be a scalar")
    if _is_zero(C):
        raise ValueError("To define a valid constraint, c must be non-zero")


def _validate_inequality_matrix(C, lower, upper):
    """Validate matrix inequality constraints (lower <= C @ x <= upper)."""
    if lower is not None and lower.dim() > 0 and C.shape[0] != lower.shape[0]:
        raise ValueError("C and lower must have matching row counts")
    if upper is not None and upper.dim() > 0 and C.shape[0] != upper.shape[0]:
        raise ValueError("C and upper must have matching row counts")
    if _is_zero(C):
        raise ValueError("To define a valid inequality constraint, C must be non-zero")


def _is_zero(X: torch.Tensor) -> bool:
    """Checks input if input tensor X is identically 0."""
    return torch.all(X == 0).item()


def _build_eval(A, C, b, lower, upper) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the constraint evaluation function.

    Constructs a function that evaluates the indicator function for the
    polyhedron defined by the provided constraints.

    Args:
        A: Equality constraint matrix (optional).
        C: Inequality constraint matrix (optional).
        b: Equality constraint vector (optional).
        lower: Lower bound vector (optional).
        upper: Upper bound vector (optional).

    Returns:
        Callable: Function that evaluates the indicator function.

    Raises:
        ValueError: If no constraints are provided.
    """
    eq_exists = A is not None and b is not None
    ineq_exists = lower is not None  # implies upper is not None

    eval_fns = []

    if eq_exists:
        if A.dim() > 1:
            eval_fns.append(partial(_eval_eq, A=A, b=b))
        else:
            eval_fns.append(partial(_eval_hyperplane, a=A, b=b))

    if ineq_exists:
        if C is not None:
            if C.dim() > 1:
                eval_fns.append(partial(_eval_ineq, C=C, lower=lower, upper=upper))
            else:
                eval_fns.append(partial(_eval_halfspace, c=C, lower=lower, upper=upper))
        else:
            eval_fns.append(partial(_eval_id_ineq, lower=lower, upper=upper))

    if not eval_fns:
        raise ValueError(
            "Provided constraints define a trivial polyhedron (no constraints)."
        )

    def _eval(x: torch.Tensor) -> torch.Tensor:
        """Evaluate all constraints.

        Args:
            x: Input vector.

        Returns:
            torch.Tensor: Sum of indicator functions (0 if all satisfied,
                inf otherwise).
        """
        return sum(fn(x) for fn in eval_fns)

    return _eval


def _indicator(
    satisfied: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return 0 if constraint satisfied, infinity otherwise.

    Args:
        satisfied: Whether the constraint is satisfied.
        device: Device for result tensor.
        dtype: Data type for result tensor.

    Returns:
        torch.Tensor: 0.0 if satisfied, infinity otherwise.
    """
    if satisfied:
        return torch.tensor(0.0, device=device, dtype=dtype)
    else:
        return torch.tensor(torch.inf, device=device, dtype=dtype)


def _eval_id_ineq(
    x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Evaluate identity inequality constraint: lower <= x <= upper."""
    satisfied = torch.all((lower <= x) & (x <= upper)).item()
    return _indicator(satisfied, x.device, x.dtype)


def _eval_ineq(
    x: torch.Tensor, C: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Evaluate matrix inequality constraint: lower <= C @ x <= upper."""
    Cx = C @ x
    satisfied = torch.all((lower <= Cx) & (Cx <= upper)).item()
    return _indicator(satisfied, x.device, x.dtype)


def _eval_halfspace(
    x: torch.Tensor, c: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Evaluate halfspace inequality constraint: lower <= c^T x <= upper."""
    cTx = torch.dot(c, x)
    satisfied = (lower <= cTx) and (cTx <= upper)
    return _indicator(satisfied.item(), x.device, x.dtype)


def _eval_eq(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate matrix equality constraint: A @ x = b."""
    satisfied = torch.all(A @ x == b).item()
    return _indicator(satisfied, x.device, x.dtype)


def _eval_hyperplane(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate hyperplane equality constraint: a^T x = b."""
    satisfied = torch.dot(a, x) == b
    return _indicator(satisfied.item(), x.device, x.dtype)
