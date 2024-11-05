from abc import ABC, abstractmethod
from jax import Array, vmap
from typing import Any, Callable

class LinearOperator(ABC):
    r"""Base interface for abstract linear operators."""

    def __init__(self, shape: tuple[int,int], ndim: int, matvec: Callable):
        r"""Initialize the linear operator.

        Args:
          shape: Shape of the linear operator.
          ndim: Dimension of the linear operator.
          _matvec: Callable that computes matvec with operator A
        """
        self.shape = shape
        self.ndim = ndim
        self._matvec = matvec

    def matvec(self, v: Array, *args: Any) -> Array:
        r"""Compute a matrix-vector product between operator A and vector v.

        Args:
          v: JAX array with matching dimension.
          *args: Optional arguments needed used to compute a matvec with A

        Returns:
          A JAX array representing Av.
        """
        return self._matvec(v, *args)
    

    def matmul(self, V: Array, *args: Any)->Array:
        return vmap(lambda v: self.matvec(v, *args), in_axes=1)(V).T
    
    def __call__(self, v, *args):
        """Performs the matrix-vector multiplication."""
        return self.matvec(v, *args)

class ExtendedLinearOperator(LinearOperator):

    def __init__(self, shape: tuple, ndim: int, matvec: Callable, rmatvec: Callable):
        super().__init__(shape, ndim, matvec)
        self._rmatvec = rmatvec
    
    
    def rmatvec(self, v: Array, *args: Any)-> Array:
        r"""Compute a matrix-vector product between the transposed operator A.T and a vector v.

        Args:
          v: JAX array with matching dimension.
          *args: Optional arguments needed used to compute a matvec with A.T

        Returns:
          A JAX array representing the resulting vector or matrix.
        """
        return self._rmatvec(v, *args)
    
    def rmatmul(self, V: Array, *args: Any)->Array:
        return vmap(lambda v: self.rmatvec(v, *args),in_axes=1)(V).T
    
    def T(self): # type: ignore
        """Returns the transpose of the linear operator as a new LinearOperator."""
        
        return ExtendedLinearOperator((self.shape[1], self.shape[0]), self.shape[1], self._rmatvec, self._matvec)
    
    @property
    def T(self):
        return self.T()