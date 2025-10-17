"""Polyhedron constraint atom for optimization."""

from functools import partial
from typing import Callable

import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable


class Polyhedron(AtomExpression):
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

    Examples:
        >>> # Box constraints: -1 <= x <= 1
        >>> x = Variable((5,), name='x')
        >>> box = Polyhedron(
        ...     x,
        ...     lower=torch.full((5,), -1.0),
        ...     upper=torch.full((5,), 1.0)
        ... )

        >>> # Equality constraint: A @ x = b
        >>> A = torch.randn(3, 5)
        >>> b = torch.randn(3)
        >>> poly = Polyhedra(x, A=A, b=b)

        >>> # Mixed constraints
        >>> C = torch.randn(2, 5)
        >>> poly = Polyhedron(
        ...     x,
        ...     A=A,
        ...     b=b,
        ...     C=C,
        ...     lower=torch.zeros(2),
        ...     upper=torch.ones(2)
        ... )
    """

    def __init__(
        self,
        x: Variable,
        A: torch.Tensor = None,
        b: torch.Tensor = None,
        C: torch.Tensor = None,
        lower: torch.Tensor = None,
        upper: torch.Tensor = None,
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
        super().__init__()
        if (A is not None) and (b is None):
            raise ValueError("b cannot be None when A is not None")

        ## Convert float/int bounds to tensors FIRST (before using .device/.dtype)
        if isinstance(lower, (int, float)):
            lower = torch.tensor(float(lower))
        if isinstance(upper, (int, float)):
            upper = torch.tensor(float(upper))

        # Validate input dimensional consistency
        _validate(A, C, b, lower, upper)

        # NOW we can safely use .device and .dtype since they're tensors
        # if upper is provided but not lower, set lower to -infinity
        if (upper is not None) and (lower is None):
            lower = torch.tensor(-torch.inf, device=upper.device, dtype=upper.dtype)
        # if lower is provided but not upper, set upper to infinity
        elif (lower is not None) and (upper is None):
            upper = torch.tensor(torch.inf, device=lower.device, dtype=lower.dtype)

        # Register the variable as a parameter
        self.register_variable(x)

        # Register constraint data as buffers
        if (A is not None) and (b is not None):
            self.register_atom_buffer("A", A)
            self.register_atom_buffer("b", b)
        else:
            self.A = None
            self.b = None

        if C is not None:
            self.register_atom_buffer("C", C)
        else:
            self.C = None

        if lower is not None:
            self.register_atom_buffer("lower", lower)
            self.register_atom_buffer("upper", upper)
        else:
            self.upper = None
            self.lower = None

        # build evaluation function
        self._eval = _build_eval(self.A, self.C, self.b, self.lower, self.upper)

    def forward(self) -> torch.Tensor:
        """Evaluate the polyhedral constraint at the current variable value.

        Returns:
            torch.Tensor: 0.0 if constraints are satisfied, infinity otherwise.
        """
        value = self.get_variable(self.var_name)
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

    def subsample(self, indices: torch.Tensor) -> "Polyhedron":
        """Subsample the polyhedral constraint (not supported).

        Args:
            indices: Indices to subsample (unused).

        Returns:
            Polyhedron: Not applicable.

        Raises:
            NotImplementedError: Polyhedron constraints cannot be subsampled.
        """
        raise NotImplementedError("Polyhedron is not subsamplable")

    def to_cvxpy(self):
        """Convert to CVXPY expression.

        Returns:
            cp.Expression: CVXPY representation (delegates to parent class).
        """
        return super().to_cvxpy()


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
        # Handle different dimensions
        if A.dim() == 0:  # Scalar A (shouldn't happen)
            raise ValueError("A must be at least 1-dimensional")

        if A.dim() == 1:  # Vector A (hyperplane: a^T x = b)
            if b.dim() != 0:  # b should be scalar
                raise ValueError("For 1D A (hyperplane), b must be a scalar")
        else:  # Matrix A (multiple constraints: A @ x = b)
            if b.dim() != 1:
                raise ValueError("For 2D A, b must be 1D")
            if A.shape[0] != b.shape[0]:
                raise ValueError("A and b must have matching row counts")

    if C is not None:
        if C.dim() == 0:
            raise ValueError("C must be at least 1-dimensional")

        if C.dim() == 1:  # Vector C (halfspace: lower <= c^T x <= upper)
            # lower and upper should be scalars
            if lower is not None and lower.dim() != 0:
                raise ValueError("For 1D C (halfspace), lower must be a scalar")
            if upper is not None and upper.dim() != 0:
                raise ValueError("For 1D C (halfspace), upper must be a scalar")
        else:  # Matrix C (multiple inequalities: lower <= C @ x <= upper)
            if lower is not None and lower.dim() > 0:
                if C.shape[0] != lower.shape[0]:
                    raise ValueError("C and lower must have matching row counts")
            if upper is not None and upper.dim() > 0:
                if C.shape[0] != upper.shape[0]:
                    raise ValueError("C and upper must have matching row counts")


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
    satisfied = torch.all((lower <= x) & (x <= upper))
    return _indicator(satisfied, x.device, x.dtype)


def _eval_ineq(
    x: torch.Tensor, C: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Evaluate matrix inequality constraint: lower <= C @ x <= upper."""
    Cx = C @ x
    satisfied = torch.all((lower <= Cx) & (Cx <= upper))
    return _indicator(satisfied, x.device, x.dtype)


def _eval_halfspace(
    x: torch.Tensor, c: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Evaluate halfspace inequality constraint: lower <= c^T x <= upper."""
    ctx = torch.dot(c, x)
    satisfied = (lower <= ctx) and (ctx <= upper)
    return _indicator(satisfied, x.device, x.dtype)


def _eval_eq(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate matrix equality constraint: A @ x = b."""
    satisfied = torch.all(A @ x == b)
    return _indicator(satisfied, x.device, x.dtype)


def _eval_hyperplane(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate hyperplane equality constraint: a^T x = b."""
    satisfied = torch.dot(a, x) == b
    return _indicator(satisfied, x.device, x.dtype)
