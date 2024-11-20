import abc

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import sketchyopts

KeyArray = Array
KeyArrayLike = ArrayLike


class LinearOperator(abc.ABC):
    r"""Base interface for abstract linear operators."""

    def __init__(self, shape: tuple, ndim: int):
        r"""Initialize the linear operator.

        Args:
          shape: Shape of the linear operator.
          ndim: Dimension of the linear operator.
        """
        self.shape = shape
        self.ndim = ndim

    @abc.abstractmethod
    def matmul(self, other: ArrayLike) -> Array:
        r"""Compute a matrix-vector or matrix-matrix product between the operator and a
        JAX array.

        Args:
          other: JAX array with matching dimension.

        Returns:
          A JAX array representing the resulting vector or matrix.
        """

    def __matmul__(self, other: ArrayLike) -> Array:
        r"""An alias for function :func:`sketchyopts.base.LinearOperator.matmul`.

        This overwrites the ``@`` operator.

        """
        return self.matmul(other)


class HessianLinearOperator(LinearOperator):
    r"""Hessian operator for computing Hessian-vector product without explicitly forming
    the Hessian matrix."""

    def __init__(
        self,
        fun,
        params,
        grad_fun=None,
        hvp_fun=None,
        *args,
        **kwargs,
    ):
        r"""Initialize the Hessian linear operator.

        The linear operator implicitly forms the Hessian of function ``fun`` with
        respect to parameters ``params``. The function can have additional positional
        and keyword arguments.

        The operator uses automatic differentiation to compute Hessian-vector product,
        unless an oracle ``hvp_fun`` is provided.

        Args:
          fun: Scalar-valued function.
          grad_fun: Optional gradient oracle.
          hvp_fun: Optional Hessian-vector product oracle.
          params: Parameters of the function.
          *args: Additional positional arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun`` if provided).
        """
        unraveled, unravel_fun = sketchyopts.util.ravel_tree(params)
        params_size = jnp.size(unraveled)

        # construct Hessian-vector product function
        if hvp_fun:
            self.hvp_fn = lambda v: sketchyopts.util.ravel_tree(
                hvp_fun(params, unravel_fun(v), *args, **kwargs)
            )[0]
        else:
            if grad_fun:
                grad_fun_partial = lambda x: sketchyopts.util.ravel_tree(
                    grad_fun(unravel_fun(x), *args, **kwargs)
                )[0]
                self.hvp_fn = lambda v: jax.jvp(grad_fun_partial, [unraveled], [v])[1]
            else:
                fun_partial = lambda x: fun(unravel_fun(x), *args, **kwargs)
                self.hvp_fn = lambda v: jax.jvp(
                    jax.grad(fun_partial), [unraveled], [v]
                )[1]

        super().__init__(shape=(params_size, params_size), ndim=2)

    def matmul(self, other):
        r"""Compute the Hessian-vector or Hessian-matrix product.

        The vector or matrix ``other`` the Hessian acts on must have the matching size
        of the parameters ``params``. Specifically, the function expects
        ``(params_size)`` for a vector or ``(params_size, num_vectors)`` for a matrix.
        If ``params`` is a pytree, the size should be the number of leaves of the tree
        (*i.e.* length of the flattened tree).

        The resulting array has size ``(params_size)`` for vector input or
        ``(params_size, num_vectors)`` for matrix input.

        Args:
          other: A 1-dimensional or 2-dimensional array with the matching size.

        Returns:
          Array representing the result.
        """
        if jnp.ndim(other) == 1:
            return self.hvp_fn(other)
        elif jnp.ndim(other) == 2:
            return jax.vmap(self.hvp_fn, 1, 1)(other)
        else:
            raise sketchyopts.errors.InputDimError("operand 1", jnp.ndim(other), [1, 2])


class AddLinearOperator(LinearOperator):
    r"""Linear operator for adding two other linear operators together."""

    def __init__(self, operator1, operator2):
        r"""Construct the linear operator.

        The function forms a new linear operator :math:`\mathrm{operator1}
        + \mathrm{operator2}`.

        Args:
          operator1: First linear operator.
          operator2: Second linear operator.
        """
        if operator1.shape != operator1.shape:
            raise ValueError("Incompatible linear operator shapes.")
        self.operator1 = operator1
        self.operator2 = operator2
        super().__init__(shape=jnp.shape(self.operator1), ndim=jnp.ndim(self.operator1))

    def matmul(self, other):
        r"""Compute the matrix-vector or matrix-matrix product of the new operator."""
        return self.operator1 @ other + self.operator2 @ other
