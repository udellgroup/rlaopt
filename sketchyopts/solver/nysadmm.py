from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax import Array

from sketchyopts.base import SolverState
from sketchyopts.nystrom import rand_nystrom_approx
from sketchyopts.operator import AddLinearOperator, HessianLinearOperator
from sketchyopts.util import (
    inexact_asarray,
    integer_asarray,
    ravel_tree,
    tree_add,
    tree_l2_norm,
    tree_ones_like,
    tree_scalar_mul,
    tree_size,
    tree_sub,
    tree_vdot,
    tree_zeros_like,
)

from .cg import abstract_cg


class NysADMMState(NamedTuple):
    r"""The NysADMM optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      res_primal: Primal residual at the current iterate.
      res_dual: Dual residual at the current iterate.
    """

    iter_num: Array
    res_primal: Array
    res_dual: Array


@dataclass(eq=False, kw_only=True)
class NysADMM:
    r"""The NysADMM optimizer.

    NysADMM is an inexact ADMM algorithm that uses randomized Nyström preconditioned
    conjugate gradients to approximately solve one of the subproblems in each iteration.
    The Nyström preconditioner is typically able to utilize the low-rank structure of
    the problem and thus speed up the iterative optimization process.

    The optimizer seeks to solve the following composite minimization problem

    .. math::

        \underset{x}{\mathrm{minimize}} ~ \Big\{\mathrm{fun}(x) + \mathrm{reg\_g}(x) + 
        \mathrm{reg\_h}(x)\Big\}

    We provide more detailed description of some arguments below. 

    .. _objective fun:
    
    ``fun`` specifies the objective function of the optimization problem. The argument 
    is **required**. It must have: 
        
    - scalar output
    - optimization variable as **first** argument
    - argument ``data`` for data input
    
    For instance, ``fun`` could take the form :python:`value = fun(params, data, 
    **fun_params)`. 

    ----

    .. _smooth regularization:
    
    ``reg_g`` specifies the smooth component of the regularization of the problem. The 
    argument is **required**. It must have: 

    - scalar output
    - optimization variable as **first** argument
    
    For instance, ``reg_g`` could take the form :python:`value = reg_g(params, 
    **reg_g_params)`. 

    ----

    .. _prox nonsmooth regularization:

    ``prox_reg_h`` specifies the proximal operator of the non-smooth component of the 
    regularization of the problem. The argument is **required**. It must have: 
        
    - point to be mapped as **first** argument
    - keyword argument ``scaling`` that sets the scaling factor of the operator
    
    More precisely, ``prox_reg_h`` solves the following optimization problem

    .. math::
       
        \mathrm{prox\_reg\_h}(x, \mathrm{scaling}) = 
        \underset{y}{\mathrm{argmin}} ~ \Big\{\mathrm{reg\_h}(x) 
        + \frac{1}{2 \cdot \mathrm{scaling}} \lVert x - y \rVert_2^2\Big\}

    For instance, ``prox_reg_h`` could take the form :python:`mapped_point = 
    prox_reg_g(point, scaling, **prox_reg_h_params)`. 

    .. note::
      SketchyOpts has some common proximal operators built in. These implementations can
      be found in the :doc:`sketchyopts.prox` module. 
    
    ----

    .. _gradient oracle:
      
    ``grad_fun`` specifies the gradient oracle that computes the gradient of the 
    objective function ``fun`` with respect to its first argument (*i.e.* the 
    optimization variable). The argument is **optional**. 

    - It is expected to have the same function signature as the objective function 
      ``fun``. For instance, ``grad_fun`` could take the form 
      :python:`grad = grad_fun(params, data, **fun_params)`.
    - The gradient output should have the same shape and type as the the optimization 
      variable. 

    ----

    .. _hvp oracle:

    ``hvp_fun`` specifies the Hessian-vector product oracle that computes the 
    product of the Hessian and an arbitrary compatible vector. It is **optional**. 
    If provided, it must have: 

    - optimization variable as **first** argument
    - arbitrary vector (of the same shape and type as the optimization variable) as 
      **second** argument
    - same additional arguments as the objective function ``fun`` (``data``, etc.)
    - function output of the same shape and type as the optimization variable

    For instance, ``hvp_fun`` could take the form :python:`vec_output = 
    hvp_fun(params, vec, data, **fun_params)`. 
    
    ----

    .. _tolerance sequence:

    ``tol_seq`` specifies the tolerance sequence for the Nyström PCG subproblem. The 
    sequence should be positive summable. By default, the optimizer uses the geometric 
    mean of  the ADMM primal residual :math:`r_p` and dual residual :math:`r_d` at the 
    previous iteration

    .. math::

        \varepsilon^{(k+1)} = \sqrt{r_p^{(k)} r_d^{(k)}}

    Alternatively, use can provide a custom function that generate such sequence. The
    function should take 

    - :math:`x`: optimization variable
    - :math:`z`: auxiliary variable
    - :math:`u`: (scaled) dual variable
    - :math:`r_p`: primal residual
    - :math:`r_d`: dual residual
    - :math:`b`: right-hand side of the Nyström PCG subproblem
    - iteration number

    as positional arguments and return a positive value. 

    ----

    .. _stopping criteria:

    ``abs_tol`` and ``rel_tol`` specify absolute and relative tolerances for terminating 
    the optimizer. Specifically, the optimization stops when 

    - the maximal number of iterations has been reached
    - or, the following stopping criteria have been met

    .. math::

        \begin{aligned}
            & \lVert r_p^{(k)} \rVert_2 \leqslant \mathrm{abs\_tol} + \mathrm{rel\_tol} 
            \cdot \max \big\{\lVert x^{(k)} \rVert_2, \lVert z^{(k)} \rVert_2\big\} \\
            & \lVert r_d^{(k)} \rVert_2 \leqslant \mathrm{abs\_tol} + \mathrm{rel\_tol} 
            \cdot \lVert \mathrm{step\_size} \, u^{(k)} \rVert_2 \\
        \end{aligned}

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.prox import prox_l1
        from sketchyopts.solver import NysADMM

        def least_squares_objective(params, data):
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        def l2_squared_reg(params, scaling):
            return (0.5 * scaling) * jnp.dot(params, params)

        reg_g_params = {'scaling': reg * (1 - l1_ratio)}
        prox_reg_h_params = {'l1reg': reg * l1_ratio}

        opt = NysADMM(fun=least_squares_objective, 
                      reg_g=l2_squared_reg, 
                      prox_reg_h=prox_l1, 
                      ...)
        params, state = opt.run(init_params, 
                                data, 
                                reg_g_params=reg_g_params, 
                                prox_reg_h_params=prox_reg_h_params)

    References:
      - S\. Zhao, Z. Frangella, and M. Udell, `NysADMM: faster composite convex
        optimization via low-rank approximation <https://proceedings.mlr.press/v162/zhao
        22a>`_, Proceedings of the 39\ :sup:`th` International Conference on Machine
        Learning, PMLR 162: 26824–26840, 2022.

    Args:
      fun: Scalar-valued objective function. :ref:`↪ More Details<objective fun>`
      reg_g: Smooth component of regularization function. :ref:`↪ More Details<smooth regularization>`
      prox_reg_h: Proximal operator of the non-smooth component of regularization function. :ref:`↪ More Details<prox nonsmooth regularization>`
      grad_fun: Optional gradient oracle corresponding to the provided objective
        function ``fun``. :ref:`↪ More Details<gradient oracle>`
      hvp_fun: Optional Hessian-vector product oracle corresponding to the provided
        objective function ``fun``. :ref:`↪ More Details<hvp oracle>`
      step_size: Step size of the optimizer. Expect a non-negative value.
      sketch_size: Rank of the Nyström preconditioner. Expect a positive value (default
        ``10``).
      update_freq: Update frequency of the preconditioner for the subproblem solver. 
        When set to ``0`` or :math:`\infty` (e.g. jax.numpy.inf or numpy.inf), the the 
        optimizer uses constant preconditioner that is constructed at the beginning of 
        the optimization process.
      tol_seq: Optional tolerance function for the subproblem. If not provided, the
        optimizer uses the geometric mean of the primal and dual residuals at the 
        previous iteration (default ``None``). :ref:`↪ More Details<tolerance sequence>`
      seed: Initial seed for the random number generator (default ``0``).
      maxiter: Maximum number of iterations to run the optimizer (default ``20``).
        Expect a positive value.
      abs_tol: Absolute criterion for terminating the optimizer (default ``1e-4``). :ref:`↪ More Details<stopping criteria>`
      rel_tol: Relative criterion for terminating the optimizer (default ``1e-4``). :ref:`↪ More Details<stopping criteria>`
      verbose: Whether to print diagnostic message (default ``False``).
      jit: Whether to JIT-compile the optimization process (default ``True``).
    """

    fun: Callable
    reg_g: Callable
    prox_reg_h: Callable

    grad_fun: Optional[Callable] = None
    hvp_fun: Optional[Callable] = None

    step_size: float
    sketch_size: int = 10
    update_freq: int = 1
    tol_seq: Optional[Callable] = None

    seed: int = 0

    maxiter: int = 20
    abs_tol: float = 1e-4
    rel_tol: float = 1e-4
    verbose: bool = False
    jit: bool = True

    def __post_init__(self):
        r"""The function constructs ``tol_seq`` if not provided."""
        if self.tol_seq is None:
            self.tol_seq = self._tol_seq_residual_geo_mean

    def _tol_seq_residual_geo_mean(self, x, z, u, r_p, r_d, b, iter_num):
        r"""The function computes the tolerance value based on the geometric mean of the
        ADMM primal and dual residuals."""
        del self, x, z, u, b, iter_num
        return jnp.minimum(jnp.sqrt(jnp.abs(tree_vdot(r_p, r_d))), 1.0)

    def _is_terminable(self, x, z, u, r_p, r_d):
        r"""The function determines if the stopping criteria are met."""
        tol_p = self.abs_tol + self.rel_tol * jnp.maximum(
            tree_l2_norm(x), tree_l2_norm(z)
        )
        tol_d = self.abs_tol + self.rel_tol * tree_l2_norm(
            tree_scalar_mul(self.step_size, u)
        )

        return (tree_l2_norm(r_p) <= tol_p) & (tree_l2_norm(r_d) <= tol_d)

    def run(
        self,
        init_params,
        data,
        fun_params={},
        reg_g_params={},
        prox_reg_h_params={},
    ) -> SolverState:
        r"""The function runs the optimization loop.

        Args:
          init_params: Initial value of the optimization variable.
          data: Full dataset. Expect an array of shape ``(num_samples, ...)``.
          fun_params: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun`` if provided).
          reg_g_params: Additional keyword arguments to be passed to ``reg_g``.
          prox_reg_h_params: Additional keyword arguments to be passed to ``prox_reg_h``.
        Returns:
          Final optimization variable and solver state. The variable has the same shape
          as the provided initial value ``init_params``, and the state is an
          :class:`NysADMM` object.
        """

        # obtain unraveling function and parameter size
        _, unravel_fun = ravel_tree(init_params)
        params_len = tree_size(init_params)

        # obtain gradient functions
        if callable(self.grad_fun):
            grad_f = lambda x: self.grad_fun(unravel_fun(x), **fun_params, data=data)  # type: ignore
        else:
            grad_f = jax.grad(
                lambda x: self.fun(unravel_fun(x), **fun_params, data=data)
            )

        grad_g = jax.grad(lambda x: self.reg_g(unravel_fun(x), **reg_g_params))

        # terminate the loop if one of the following holds:
        # - maximal number of iterations has been reached
        # - stopping criteria have been met
        def cond_fun(value):
            x, z, u, r_p, r_d, _, _, iter_num, _ = value
            return (iter_num < self.maxiter) & (~self._is_terminable(x, z, u, r_p, r_d))

        def body_fun(value):
            x, z, u, r_p, r_d, U, S, iter_num, key = value

            # unravel iterate x
            x_unraveled, _ = ravel_tree(x)
            z_unraveled, _ = ravel_tree(z)
            u_unraveled, _ = ravel_tree(u)

            # form linear system
            H_f = HessianLinearOperator(
                fun=self.fun,
                grad_fun=self.grad_fun,
                hvp_fun=self.hvp_fun,
                params=x,
                **fun_params,
                data=data,
            )
            H_g = HessianLinearOperator(
                fun=self.reg_g,
                grad_fun=None,
                hvp_fun=None,
                params=x,
                **reg_g_params,
            )
            A = AddLinearOperator(H_f, H_g)
            b = (
                self.step_size * (z_unraveled - u_unraveled)
                + H_f @ x_unraveled
                + H_g @ x_unraveled
                - grad_f(x_unraveled)
                - grad_g(x_unraveled)
            )

            # x step
            pcg_tol = self.tol_seq(x, z, u, r_p, r_d, b, iter_num)  # type: ignore

            def update_preconditioner():
                _key, nystrom_key = jax.random.split(key)  # type: ignore
                _U, _S = rand_nystrom_approx(A, self.sketch_size, nystrom_key)
                return _U, _S, _key

            def keep_preconditioner():
                return U, S, key

            if self.update_freq == 1:
                U, S, key = update_preconditioner()
            else:
                if self.update_freq == 0:
                    update_iter = eqxi.unvmap_max(iter_num) == 0
                else:
                    update_iter = (eqxi.unvmap_max(iter_num) % self.update_freq) == 0
                update_iter = eqxi.nonbatchable(update_iter)
                U, S, key = jax.lax.cond(
                    update_iter, update_preconditioner, keep_preconditioner
                )

            def M(v):
                UTv = U.T @ v
                return (
                    (S[-1] + self.step_size)
                    * U
                    @ (UTv / jnp.expand_dims(S + self.step_size, axis=1))
                    + v
                    - U @ UTv
                )

            x, _, _, _, _ = abstract_cg(
                A,
                jnp.expand_dims(b, axis=1),
                self.step_size,
                jnp.expand_dims(x_unraveled, axis=1),
                pcg_tol,
                params_len * 10,
                M,
            )
            x = unravel_fun(jnp.ravel(x))

            # z step
            z_ = self.prox_reg_h(
                tree_add(x, u), **prox_reg_h_params, scaling=1.0 / self.step_size
            )

            # u step
            u = tree_add(u, tree_sub(x, z_))

            # compute residuals
            r_p = tree_sub(x, z_)
            r_d = tree_scalar_mul(self.step_size, tree_sub(z, z_))

            return x, z_, u, r_p, r_d, U, S, iter_num + 1, key

        # initialization
        initial_value = (
            init_params,
            tree_zeros_like(init_params),
            tree_zeros_like(init_params),
            tree_scalar_mul(jnp.inf, tree_ones_like(init_params)),
            tree_scalar_mul(jnp.inf, tree_ones_like(init_params)),
            inexact_asarray(jnp.zeros((params_len, self.sketch_size))),
            inexact_asarray(jnp.zeros((self.sketch_size,))),
            integer_asarray(0),
            jax.random.PRNGKey(self.seed),
        )

        # perform iterative solve
        if self.jit:
            x, z, u, r_p, r_d, _, _, iter_num, _ = jax.lax.while_loop(
                cond_fun, body_fun, initial_value
            )
        else:
            value = initial_value
            while cond_fun(value):
                value = body_fun(value)
            x, z, u, r_p, r_d, _, _, iter_num, _ = value

        # print out warning message if needed
        if not self._is_terminable(x, z, u, r_p, r_d):
            print(
                "Warning: run was terminated because the maximum number of iterations has been reached."
            )

        return SolverState(
            params=x,
            state=NysADMMState(iter_num=iter_num, res_primal=r_p, res_dual=r_d),
        )
