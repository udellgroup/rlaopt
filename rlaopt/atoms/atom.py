"""Base class for optimization atoms."""

from abc import ABC, abstractmethod
import torch
import cvxpy as cp
from typing import Optional


class Atom(torch.nn.Module, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable, etc.) and can be composed
    to form more complex objective functions.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Evaluates the atom and returns its value as a tensor."""
        pass

    @abstractmethod
    def is_smooth(self) -> bool:
        """Returns True if the atom is smooth (differentiable everywhere)."""
        pass

    def gradient(self) -> Optional[torch.Tensor]:
        """Returns the gradient if the atom is smooth, None otherwise.

        Default implementation uses PyTorch autograd for smooth atoms. Non-smooth atoms
        should return None.
        """
        if not self.is_smooth():
            raise NotImplementedError(
                f"{self.__class__.__name__} is non-smooth and does "
                "not support gradient computation"
            )
        raise NotImplementedError("Gradient computation not implemented for this atom")

    @abstractmethod
    def is_proxable(self) -> bool:
        """Returns True if the atom has a computable proximal operator."""
        pass

    def prox(self, location: torch.Tensor) -> torch.Tensor:
        """Proximal operator of the atom.

        Args:
            location: Point at which to evaluate the proximal operator
            step_size: Step size parameter (lambda in prox_{lambda*f})

        Returns:
            Result of the proximal operator

        Raises:
            NotImplementedError: If the atom is not proxable
        """
        if not self.is_proxable():
            raise NotImplementedError(f"{self.__class__.__name__} is not proxable")
        raise NotImplementedError("Proximal operator not implemented")

    @abstractmethod
    def is_subsamplable(self) -> bool:
        """Returns True if the atom supports subsampling (e.g., for stochastic
        methods)."""
        pass

    def subsample(self, indices: torch.Tensor) -> "Atom":
        """Returns a subsampled version of the atom.

        Args:
            indices: Indices to subsample

        Returns:
            New atom representing the subsampled version

        Raises:
            NotImplementedError: If the atom is not subsamplable
        """
        if not self.is_subsamplable():
            raise NotImplementedError(f"{self.__class__.__name__} is not subsamplable")
        raise NotImplementedError("Subsampling not implemented")

    def to_cvxpy(self) -> cp.Expression:
        """Converts the atom to a CVXPY expression for convex optimization.

        Returns:
            CVXPY expression representing this atom

        Raises:
            NotImplementedError: If conversion to CVXPY is not supported
        """
        raise NotImplementedError("CVXPY conversion not implemented")

    # Operator overloading for composition
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


class SumAtom(Atom):
    """Sum of multiple atoms."""

    def __init__(self, atoms: list[Atom]):
        super().__init__()
        self.atoms = torch.nn.ModuleList(atoms)

    def forward(self) -> torch.Tensor:
        return sum(atom.forward() for atom in self.atoms)

    def is_smooth(self) -> bool:
        return all(atom.is_smooth() for atom in self.atoms)

    def is_proxable(self) -> bool:
        # Sum is proxable if it has at most one non-smooth term
        non_smooth_count = sum(1 for atom in self.atoms if not atom.is_smooth())
        return non_smooth_count <= 1 and all(atom.is_proxable() for atom in self.atoms)

    def __mul__(self, scalar: float) -> "SumAtom":
        """Scale all atoms in the sum."""
        return SumAtom([atom * scalar for atom in self.atoms])

    def __rmul__(self, scalar: float) -> "SumAtom":
        """Scale all atoms in the sum (reverse multiplication)."""
        return self.__mul__(scalar)

    def to_cvxpy(self) -> cp.Expression:
        return sum(atom.to_cvxpy() for atom in self.atoms)
