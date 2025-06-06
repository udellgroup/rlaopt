"""Base class for optimization atoms."""

from abc import ABC, abstractmethod
import torch
import cvxpy as cp


class Atom(torch.nn.Module, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable, etc.) and can be composed
    to form more complex objective functions.
    """

    def __init__(self, scaling: float = 1.0):
        """Initializes the atom with an optional scaling factor."""
        super().__init__()
        self.scaling = scaling

    @abstractmethod
    def _forward_impl(self, location: torch.Tensor) -> torch.Tensor:
        """Unscaled evaluation of the atom."""
        pass

    def forward(self, location: torch.Tensor) -> torch.Tensor:
        """Evaluates the atom and returns its value as a tensor.

        Args:
            location: Point at which to evaluate the atom

        Returns:
            Value of the atom at the specified location
        """
        return self.scaling * self._forward_impl(location)

    @abstractmethod
    def is_smooth(self) -> bool:
        """Returns True if the atom is smooth (differentiable everywhere)."""
        pass

    def gradient(
        self, location: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        """Returns the gradient of the atom, computed with automatic differentiation.

        Args:
            location: Point at which to evaluate the gradient
            create_graph: If True, will create a graph for the gradient computation.
            This can be useful for computing higher-order derivatives.

        Returns:
            Gradient of the atom at the specified location

        Raises:
            NotImplementedError: If the atom is not smooth (gradient is not defined).
            ValueError: If the output of the atom's forward method is not a scalar.
        """
        if not self.is_smooth():
            raise NotImplementedError("Gradient is not defined for non-smooth atoms.")

        # Ensure we can compute gradients
        # If not, detach and clone the tensor to ensure it requires gradients
        if not location.requires_grad:
            location = location.detach().clone().requires_grad_(True)

        output = self.forward(location)

        if output.numel() != 1:
            raise ValueError(
                "Gradient can only be computed for scalar outputs. "
                "Please ensure the atom's forward method returns a scalar."
            )

        grad = torch.autograd.grad(
            outputs=self.forward(location), inputs=location, create_graph=create_graph
        )[0]

        return grad

    @abstractmethod
    def is_proxable(self) -> bool:
        """Returns True if the atom has a computable proximal operator."""
        pass

    @abstractmethod
    def prox(self, location: torch.Tensor) -> torch.Tensor:
        """Proximal operator of the atom.

        This method should only be called if the atom is proxable. Otherwise, it should
        raise a NotImplementedError.

        Note that this function has to account for the scaling factor of the atom.

        Args:
            location: Point at which to evaluate the proximal operator

        Returns:
            Result of the proximal operator
        """
        pass

    @abstractmethod
    def is_subsamplable(self) -> bool:
        """Returns True if the atom supports subsampling (e.g., for stochastic
        methods)."""
        pass

    @abstractmethod
    def subsample(self, indices: torch.Tensor) -> "Atom":
        """Returns a subsampled version of the atom.

        This method should only be called if the atom is subsamplable.
        Otherwise, it should raise a NotImplementedError.

        Args:
            indices: Indices to subsample

        Returns:
            New atom representing the subsampled version

        Raises:
            NotImplementedError: If the atom does not support subsampling.
        """
        pass

    @abstractmethod
    def to_cvxpy(self, variable_or_expr: cp.Variable | cp.Expression) -> cp.Expression:
        """Converts the atom to a CVXPY expression.

        This method can be called with either a CVXPY variable (for initial atoms)
        or a CVXPY expression (for atoms used in composition).

        Args:
            variable_or_expr: Either a cp.Variable or a cp.Expression

        Returns:
            CVXPY expression representing this atom
        """
        pass

    @abstractmethod
    def __mul__(self, scalar: float) -> "Atom":
        """Scale the atom by a scalar.

        Each atom handles its own scaling.
        """
        pass

    def __rmul__(self, scalar: float) -> "Atom":
        """Scale the atom by a scalar (reverse multiplication)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Atom":
        """Divide the atom by a scalar."""
        return self.__mul__(1.0 / scalar)

    def __add__(self, other: "Atom") -> "SumAtom":
        """Add atoms together."""
        if isinstance(other, Atom):
            return SumAtom([self, other])
        return NotImplemented

    def __radd__(self, other: "Atom") -> "SumAtom":
        """Add atoms together (reverse addition)."""
        return self.__add__(other)

    def __sub__(self, other: "Atom") -> "SumAtom":
        """Subtract atoms (self - other)."""
        if isinstance(other, Atom):
            return SumAtom([self, other * (-1.0)])
        return NotImplemented

    def __rsub__(self, other: "Atom") -> "SumAtom":
        """Reverse subtraction (other - self)."""
        if isinstance(other, Atom):
            return other.__sub__(self)
        return NotImplemented


class SumAtom(Atom):
    """Sum of multiple atoms."""

    def __init__(self, atoms: list[Atom]):
        super().__init__()

        # Flatten nested SumAtoms during initialization
        flattened_atoms = []
        for atom in atoms:
            if isinstance(atom, SumAtom):
                flattened_atoms.extend(list(atom.atoms))
            else:
                flattened_atoms.append(atom)

        self.atoms = torch.nn.ModuleList(flattened_atoms)

    def forward(self, location: torch.Tensor) -> torch.Tensor:
        return sum(atom.forward(location) for atom in self.atoms)

    def is_smooth(self) -> bool:
        return all(atom.is_smooth() for atom in self.atoms)

    def is_proxable(self) -> bool:
        # Default to False - subclasses should override with specific logic
        return False

    def prox(self, location: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SumAtom does not have a prox operator by default.")

    def is_subsamplable(self) -> bool:
        # Default to False - subclasses should override with specific logic
        return False

    def subsample(self, indices: torch.Tensor) -> "SumAtom":
        raise NotImplementedError("SumAtom does not support subsampling by default.")

    def __mul__(self, scalar: float) -> "SumAtom":
        """Scale all atoms in the sum."""
        return SumAtom([atom * scalar for atom in self.atoms])

    def to_cvxpy(self, variable_or_expr: cp.Variable | cp.Expression) -> cp.Expression:
        return sum(atom.to_cvxpy(variable_or_expr) for atom in self.atoms)


class ComposedAtom(Atom):
    """Represents function composition f_n(f_{n-1}(...f_1(x)))

    This atom represents the mathematical composition of multiple functions, where each
    function is applied in sequence.
    """

    def __init__(self, atoms: list[Atom], scaling: float = 1.0):
        """Initialize a composed atom.

        Args:
            atoms: List of atoms to compose, in order of application
            (first atom is applied first)
            scaling: Optional scaling factor for the entire composition
        """
        super().__init__(scaling=scaling)

        # Flatten nested ComposedAtoms during initialization
        flattened_atoms = []
        for atom in atoms:
            if isinstance(atom, ComposedAtom):
                # For nested ComposedAtom, maintain the order of composition
                flattened_atoms.extend(list(atom.atoms))
            else:
                flattened_atoms.append(atom)

        self.atoms = torch.nn.ModuleList(flattened_atoms)

    def _forward_impl(self, location: torch.Tensor) -> torch.Tensor:
        """Evaluates the composition by applying each atom in sequence."""
        result = location
        for atom in self.atoms:
            result = atom.forward(result)
        return result

    def is_smooth(self) -> bool:
        """A composition is smooth if all constituent atoms are smooth."""
        return all(atom.is_smooth() for atom in self.atoms)

    def is_proxable(self) -> bool:
        """In general, composed atoms are not proxable."""
        return False

    def prox(self, location: torch.Tensor) -> torch.Tensor:
        """Proximal operator for composed functions."""
        raise NotImplementedError(
            "Proximal operator not available for general function composition"
        )

    def is_subsamplable(self) -> bool:
        """In general, composed atoms are not subsamplable."""
        return False

    def subsample(self, indices: torch.Tensor) -> "ComposedAtom":
        """Creates a subsampled version of this composition."""
        raise NotImplementedError(
            "ComposedAtom does not support subsampling by default."
        )

    def to_cvxpy(self, variable_or_expr) -> cp.Expression:
        """Converts the atom to a CVXPY expression for convex optimization.

        This method handles function composition by sequentially applying each atom's
        to_cvxpy method to the result of the previous atom.

        Args:
            variable_or_expr: Either a cp.Variable or a cp.Expression to use as input

        Returns:
            CVXPY expression representing this composed atom

        Raises:
            ValueError: If there are no atoms
        """
        # Handle the case where there are no atoms
        if len(self.atoms) == 0:
            raise ValueError("ComposedAtom must contain at least one atom.")

        # Start with the input variable or expression
        expr = variable_or_expr

        # Apply each atom in sequence
        for i, atom in enumerate(self.atoms):
            # Apply the current atom to the result of the previous step
            expr = atom.to_cvxpy(expr)

        # Apply scaling to the final expression
        return self.scaling * expr

    def __mul__(self, scalar: float) -> "ComposedAtom":
        """Scale the composed atom."""
        return ComposedAtom(list(self.atoms), scaling=self.scaling * scalar)
