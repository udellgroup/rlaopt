from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import sparse
from jax.typing import ArrayLike

from sketchyopts.base import (
    AddLinearOperator,
    HessianLinearOperator,
    LinearOperator,
    PromiseSolver,
    SolverState,
)
from sketchyopts.errors import InputDimError, MatrixNotSquareError
from sketchyopts.preconditioner import rand_nystrom_approx
from sketchyopts.util import (
    inexact_asarray,
    integer_asarray,
    ravel_tree,
    tree_add,
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_scalar_mul,
    tree_sub,
    tree_vdot,
    tree_zeros_like,
)

KeyArray = Array
KeyArrayLike = ArrayLike


def nystrom_pcg(
    A: Any,
    b: ArrayLike,
    mu: float,
    rank: int,
    key: KeyArrayLike,
    *,
    x0: Optional[ArrayLike] = None,
    maxiter: Optional[int] = None,
    tol: float = 1e-5,
) -> tuple[Array, Array, Array, int]:
    r"""The Nyström preconditioned conjugate gradient method (Nyström PCG).

    The function solves the regularized linear system :math:`(A + \mu I)x = b` using
    Nyström PCG.

    Nyström PCG uses randomized Nyström preconditioner by implicitly applying

    .. math::
        P^{-1}
        = (\hat{\lambda}_{l} + \mu) U (\hat{\Lambda} + \mu I)^{-1} U^{T} + (I - U U^{T})

    where :math:`U` and :math:`\hat{\Lambda}` are from rank-:math:`l` randomized Nyström
    approximation (here :math:`\hat{\lambda}_{l}` is the :math:`l`:sup:`th` diagonal
    entry of :math:`\hat{\Lambda}`).

    Nyström PCG terminates if the :math:`\ell^2`-norm of the residual
    :math:`b - (A + \mu I)\hat{x}` is within the specified tolerance or it has reached
    the maximal number of iterations.

    References:
      - Z\. Frangella, J. A. Tropp, and M. Udell, `Randomized Nyström preconditioning
        <https://epubs.siam.org/doi/10.1137/21M1466244>`_. SIAM Journal on Matrix
        Analysis and Applications, vol. 44, iss. 2, 2023, pp. 718-752.

    Args:
      A: A two-dimensional array representing a positive-semidefinite matrix.
      b: A vector or a two-dimensional array giving the righthand side(s) of the
        regularized linear system.
      mu: Regularization parameter. Expect a non-negative value.
      rank: Rank of the randomized Nyström approximation (which coincides with sketch
        size). Expect a positive value.
      key: A PRNG key used as the random key.
      x0: Initial guess for the solution (same size as righthand side(s) ``b``; default
        ``None``). When set to ``None``, the algorithm uses zero vector as starting
        guess.
      maxiter: Maximum number of iterations (default ``None``). When set to ``None``,
        the algorithm only terminates when the specified tolerance has been achieved.
        Internally the value gets set to :math:`10` times the size of the system.
      tol: Solution tolerance (default ``1e-5``).

    Returns:
      A four-element tuple containing

      - **x** – Approximate solution to the regularized linear system. Solution has the
        same size as righthand side(s) ``b``.
      - **r** – Residual of the approximate solution. Residual has the same size as
        righthand side(s) ``b``.
      - **status** – Whether or not the approximate solution has converged for each
        righthand side. Status has the same size as the number of righthand side(s).
      - **k** – Total number of iterations to reach to the approximate solution.
    """
    # perform randomized Nyström approximation
    U, S = rand_nystrom_approx(A, rank, key)

    # matrix-vector (or mat-mat for multiple righthand sides) product for regularized
    # linear operator
    @jax.jit
    def regularized_A(x):
        return A @ x + mu * x

    # matrix-vector (or mat-mat for multiple righthand sides) product for inverse
    # Nyström preconditioner
    @jax.jit
    def inv_preconditioner(x):
        UTx = U.T @ x
        return (S[-1] + mu) * U @ (UTx / jnp.expand_dims(S + mu, axis=1)) + x - U @ UTx

    # condition evaluation
    def cond_fun(value):
        _, _, _, _, mask, k = value
        return (jnp.sum(mask) > 0) & (k < maxiter)

    # PCG iteration
    def body_fun(value):
        x, r, z, p, mask, k = value
        # select only columns corresponding to the righthand side that has yet converged
        # populate the remaining columns with NaN
        xs = jnp.where(mask > 0, x[:,], jnp.nan)
        rs = jnp.where(mask > 0, r[:,], jnp.nan)
        zs = jnp.where(mask > 0, z[:,], jnp.nan)
        ps = jnp.where(mask > 0, p[:,], jnp.nan)
        # perform update on selected columns and ignore padded columns
        # (i.e. with NaN values)
        v = regularized_A(ps)
        gamma = jnp.sum(rs * zs, axis=0, keepdims=True)  # type: ignore
        alpha = gamma / jnp.sum(ps * v, axis=0, keepdims=True)
        x_ = jnp.where(mask > 0, (xs + alpha * ps)[:,], x[:,])
        r_s = rs - alpha * v
        r_ = jnp.where(mask > 0, r_s[:,], r[:,])
        z_s = inv_preconditioner(r_s)
        z_ = jnp.where(mask > 0, z_s[:,], z[:,])
        beta = jnp.sum(r_s * z_s, axis=0, keepdims=True) / gamma
        p_ = jnp.where(mask > 0, (z_s + beta * ps)[:,], p[:,])
        r_norm = jnp.linalg.norm(r_s, axis=0)
        mask_ = jnp.where(r_norm > tol, 1, 0)  # NaN always evaluates to False
        return x_, r_, z_, p_, mask_, k + 1

    # dimension check
    if jnp.ndim(A) != 2:
        raise InputDimError("A", jnp.ndim(A), 2)
    else:
        if jnp.shape(A)[0] != jnp.shape(A)[1]:
            raise MatrixNotSquareError("A", jnp.shape(A))

    if jnp.ndim(b) not in [1, 2]:
        raise InputDimError("b", jnp.ndim(b), [1, 2])

    # initialization
    b_ndim = jnp.ndim(b)
    if b_ndim == 1:
        b = jnp.expand_dims(b, axis=1)
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if maxiter is None:
        maxiter = jnp.shape(b)[0] * 10  # same behavior as SciPy

    # initial step
    r0 = b - regularized_A(x0)
    p0 = z0 = inv_preconditioner(r0)
    mask0 = jnp.ones(jnp.shape(b)[1], dtype=int)
    initial_value = (x0, r0, z0, p0, mask0, 0)

    # perform iterative solve
    x_final, r_final, _, _, mask_final, k_final = jax.lax.while_loop(
        cond_fun, body_fun, initial_value
    )

    # match solution and residual to the input shape if b has a single dimension
    if b_ndim == 1:
        x_final = jnp.squeeze(x_final)  # type: ignore
        r_final = jnp.squeeze(r_final)

    return x_final, r_final, ~mask_final.astype(bool), k_final  # type: ignore


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

    - the Nyström PCG has failed to solve the subproblem to the required tolerance 
      (based on the tolerance sequence)
    - or, the maximal number of iterations has been reached
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

        opt = NysADMM(least_squares_objective, l2_squared_reg, prox_l1, ...)
        opt.run(init_params, data, reg_g_params=reg_g_params, 
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
        return jnp.sqrt(tree_vdot(r_p, r_d))

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

        # obtain unraveling function
        _, unravel_fun = ravel_tree(init_params)

        # obtain gradient functions
        if callable(self.grad_fun):
            grad_f = lambda x: self.grad_fun(unravel_fun(x), **fun_params, data=data)  # type: ignore
        else:
            grad_f = jax.grad(
                lambda x: self.fun(unravel_fun(x), **fun_params, data=data)
            )

        grad_g = jax.grad(lambda x: self.reg_g(unravel_fun(x), **reg_g_params))

        # terminate the loop if one of the following holds:
        # - x step PCG did not converge to the tolerance
        # - maximal number of iterations has been reached
        # - stopping criteria have been met
        def cond_fun(value):
            x, z, u, r_p, r_d, iter_num, pcg_status, _ = value
            return (
                pcg_status[0]
                & (iter_num < self.maxiter)
                & (~self._is_terminable(x, z, u, r_p, r_d))
            )

        def body_fun(value):
            x, z, u, r_p, r_d, iter_num, _, key = value

            # unravel iterate x
            x_unravaled, _ = ravel_tree(x)
            z_unravaled, _ = ravel_tree(z)
            u_unravaled, _ = ravel_tree(u)

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
                self.step_size * (z_unravaled - u_unravaled)
                + H_f @ x_unravaled
                + H_g @ x_unravaled
                - grad_f(x_unravaled)
                - grad_g(x_unravaled)
            )

            # x step
            key, pcg_key = jax.random.split(key)
            pcg_tol = self.tol_seq(x, z, u, r_p, r_d, b, iter_num)  # type: ignore
            x, _, pcg_status, _ = nystrom_pcg(
                A,
                b,
                self.step_size,
                self.sketch_size,
                pcg_key,
                x0=jnp.expand_dims(x_unravaled, axis=1),
                maxiter=None,
                tol=pcg_tol,
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

            return x, z_, u, r_p, r_d, iter_num + 1, pcg_status, key

        # initialization
        x = init_params
        z = tree_zeros_like(init_params)
        u = tree_zeros_like(init_params)
        r_p = tree_scalar_mul(jnp.inf, tree_zeros_like(init_params))
        r_d = tree_scalar_mul(jnp.inf, tree_zeros_like(init_params))
        key = jax.random.PRNGKey(self.seed)

        initial_value = (
            x,
            z,
            u,
            r_p,
            r_d,
            integer_asarray(0),
            jnp.ones(1).astype(bool),
            key,
        )

        # perform iterative solve
        if self.jit:
            x, z, u, r_p, r_d, iter_num, pcg_status, _ = jax.lax.while_loop(
                cond_fun, body_fun, initial_value
            )
        else:
            value = initial_value
            while cond_fun(value):
                value = body_fun(value)
            x, z, u, r_p, r_d, iter_num, pcg_status, _ = value

        # print out warning message if needed
        if not self._is_terminable(x, z, u, r_p, r_d):
            if not pcg_status:
                print(
                    "Warning: run was terminated because linear solve has failed to converge the tolerance."
                )
            else:
                print(
                    "Warning: run was terminated because the maximum number of iterations has been reached."
                )

        return SolverState(
            params=x,
            state=NysADMMState(iter_num=iter_num, res_primal=r_p, res_dual=r_d),
        )


class SketchySGDState(NamedTuple):
    r"""The SketchySGD optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      step_size: Step size for the next update.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    step_size: Array


@dataclass(eq=False, kw_only=True)
class SketchySGD(PromiseSolver):
    r"""The SketchySGD optimizer.

    SketchySGD is a stochastic quasi-Newton method that uses sketching to approximate
    the curvature of the loss function. It maintains a preconditioner for SGD
    (stochastic gradient descent) using subsampled Hessian and automatically selects an
    appropriate learning whenever it updates the preconditioner.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchySGD

        def ridge_reg_objective(params, reg, data):
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchySGD(ridge_reg_objective, ...)
        opt.run(init_params, data, reg)

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `SketchySGD: Reliable
        Stochastic Optimization via Randomized Curvature Estimates
        <https://arxiv.org/abs/2211.08597>`_.

    Args:
      fun: Scalar-valued objective function. :term:`↪ More Details<fun>`
      grad_fun: Optional gradient oracle corresponding to the provided objective
        function ``fun``. :term:`↪ More Details<grad_fun>`
      hvp_fun: Optional Hessian-vector product oracle for the Nyström subsampled Newton
        preconditioner. :term:`↪ More Details<hvp_fun>`
      sqrt_hess_fun: Required oracle that computes the square root of the Hessian matrix
        for the subsampled Newton preconditioner (*i.e.* cannot be empty if ``precond``
        is set to ``ssn``). :term:`↪ More Details<sqrt_hess_fun>`
      pre_update: Optional function to execute before optimizer's each update on the
        iterate. :term:`↪ More Details<pre_update>`
      precond: Type of preconditioner to use. Either ``nyssn`` for Nyström subsampled
        Newton or ``ssn`` for subsampled Newton (default ``nyssn``).
      rho: Regularization parameter for the preconditioner. Expect a non-negative value
        (default ``1e-3``).
      rank: Rank of the Nyström subsampled Newton preconditioner. Expect a positive
        value (default ``10``).
      grad_batch_size: Size of the batch of data to compute stochastic gradient at each
        iteration. Expect a positive value.
      hess_batch_size: Size of the batch of data to estimate the stochastic Hessian when
        updating the preconditioner. Expect a positive value.
      update_freq: Update frequency of the preconditioner. When set to ``0`` or
        :math:`\infty` (*e.g.* ``jax.numpy.inf`` or ``numpy.inf``), the optimizer uses
        constant preconditioner that is constructed at the beginning of the optimization
        process.
      seed: Initial seed for the random number generator (default ``0``).
      learning_rate: Step size for applying updates (default ``0.5``). It can either be
        a fixed scalar value or a schedule (callable) based on step count.
        :term:`↪ More Details<learning_rate>`
      maxiter: Maximum number of iterations to run the optimizer (default ``20``).
        Expect a positive value.
      tol: Threshold of the gradient norm used for terminating the optimizer (default
        ``1e-3``).
      verbose: Whether to print diagnostic message (default ``False``).
      jit: Whether to JIT-compile the optimization process (default ``True``).
        :term:`↪ More Details<jit>`
      sparse: Whether to sparsify the optimization process (default ``False``).
        :term:`↪ More Details<sparse>`
    """

    learning_rate: Union[float, Callable] = 0.5

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchySGDState:
        r"""The function initializes the optimizer state."""
        return SketchySGDState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            step_size=inexact_asarray(0.0),
        )

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.grad_batch_size,), replace=False
        )

        # compute preconditioned stochastic gradient
        value, grad = self._value_and_grad(
            params,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )
        error = tree_l2_norm(grad, squared=False)
        unraveled_grad, unravel_fun = ravel_tree(grad)
        direction = unravel_fun(self._grad_transform(unraveled_grad, state.precond))

        # perform an update
        params = tree_add_scalar_mul(params, -state.step_size, direction)

        return SolverState(
            params=params,
            state=state._replace(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
            ),
        )

    def run(
        self, init_params: Any, data: ArrayLike, reg: float = 0.0, *args, **kwargs
    ) -> SolverState:
        r"""The function runs the optimization loop.

        Args:
          init_params: Initial value of the optimization variable.
          data: Full dataset. Expect an array of shape ``(num_samples, ...)``.
          reg: Regularization strength. Expect a non-negative value (default ``0``).
          *args: Additional positional arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
        Returns:
          Final optimization variable and solver state. The variable has the same shape
          as the provided initial value ``init_params``, and the state is an
          :class:`SketchySGDState` object.
        """
        # initialize iterate and state
        params = init_params
        self.num_samples = jnp.shape(data)[0]
        self.params_len = jnp.size(ravel_tree(params)[0])
        state = self._init_state(params, data, reg, *args, **kwargs)

        # define partial functions
        update_params = lambda p, s: self._update_params(
            p, s, data, reg, *args, **kwargs
        )
        update_precond = lambda p, s: self._update_precond(
            p, s, data, reg, *args, **kwargs
        )

        # sparsify functions if needed
        if self.sparse:
            update_params = sparse.sparsify(update_params)
            update_precond = sparse.sparsify(update_precond)

        # JIT-compile functions if needed
        if self.jit:
            update_params = (
                jax.jit(update_params) if self.maxiter > 1 else update_params
            )
            update_precond = (
                jax.jit(update_precond)
                if (self.update_freq > 0 and self.maxiter / self.update_freq > 1)
                else update_precond
            )

        # run the optimization loop
        for i in range(self.maxiter):
            # call custom pre-update function
            if self.pre_update:
                params, state = self.pre_update(
                    params, state, *args, **kwargs, data=data, reg=reg
                )

            # update preconditioner
            if (self.update_freq == 0 and state.iter_num == 0) or (
                self.update_freq > 0 and state.iter_num % self.update_freq == 0
            ):
                state = update_precond(params, state)

            # update iterate
            params, state = update_params(params, state)

            # break out of loop if tolerance has been reached
            if state.error < self.tol:
                print(
                    "Info: early termination because error tolerance has been reached."
                )
                break

        return SolverState(params=params, state=state)


class SketchySVRGState(NamedTuple):
    r"""The SketchySVRG optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      full_grad: Full gradient at the snapshot.
      snapshot: Snapshot of the iterate.
      step_size: Step size for the next update.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    full_grad: Array
    snapshot: Array
    step_size: Array


@dataclass(eq=False, kw_only=True)
class SketchySVRG(PromiseSolver):
    r"""The SketchySVRG optimizer.

    SketchySVRG is a preconditioned version of SVRG [#f1]_ (stochastic variance reduced
    gradient). The optimizer stores and periodically updates a snapshot of the iterate,
    and uses the snapshot and its full gradient to perform variance reduction. The
    preconditioning is then applied to the variance-reduced stochastic gradient at each
    step. The preconditioner and learning rate get updated periodically.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchySVRG

        def ridge_reg_objective(params, reg, data):
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchySVRG(ridge_reg_objective, ...)
        opt.run(init_params, data, reg)

    .. rubric:: References

    .. [#f1] R\. Johnson and T. Zhang, `Accelerating Stochastic Gradient Descent using
      Predictive Variance Reduction <https://papers.nips.cc/paper/4937-accelerating-
      stochastic-gradient-descent-using-predictive-variance-reduction>`_, Advances in
      Neural Information Processing Systems, 26, 2013.

    - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned
      Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates
      <https://arxiv.org/abs/2309.02014>`_.

    Args:
      fun: Scalar-valued objective function. :term:`↪ More Details<fun>`
      grad_fun: Optional gradient oracle corresponding to the provided objective
        function ``fun``. :term:`↪ More Details<grad_fun>`
      hvp_fun: Optional Hessian-vector product oracle for the Nyström subsampled Newton
        preconditioner. :term:`↪ More Details<hvp_fun>`
      sqrt_hess_fun: Required oracle that computes the square root of the Hessian matrix
        for the subsampled Newton preconditioner (*i.e.* cannot be empty if ``precond``
        is set to ``ssn``). :term:`↪ More Details<sqrt_hess_fun>`
      pre_update: Optional function to execute before optimizer's each update on the
        iterate. :term:`↪ More Details<pre_update>`
      precond: Type of preconditioner to use (default ``nyssn``). Either ``nyssn`` for
        Nyström subsampled Newton or ``ssn`` for subsampled Newton.
        :term:`↪ More Details<precond>`
      rho: Regularization parameter for the preconditioner. Expect a non-negative value
        (default ``1e-3``).
      rank: Rank of the Nyström subsampled Newton preconditioner. Expect a positive
        value (default ``10``).
      grad_batch_size: Size of the batch of data to compute stochastic gradient at each
        iteration. Expect a positive value.
      hess_batch_size: Size of the batch of data to estimate the stochastic Hessian when
        updating the preconditioner. Expect a positive value.
      update_freq: Update frequency of the preconditioner. When set to ``0`` or
        :math:`\infty` (*e.g.* ``jax.numpy.inf`` or ``numpy.inf``), the optimizer uses
        constant preconditioner that is constructed at the beginning of the optimization
        process.
      snapshop_update_freq: Update frequency of the snapshot. Expect a positive value.
      seed: Initial seed for the random number generator (default ``0``).
      learning_rate: Step size for applying updates (default ``0.5``). It can either be
        a fixed scalar value or a schedule (callable) based on step count.
        :term:`↪ More Details<learning_rate>`
      maxiter: Maximum number of iterations to run the optimizer (default ``20``).
        Expect a positive value.
      tol: Threshold of the gradient norm used for terminating the optimizer (default
        ``1e-3``).
      verbose: Whether to print diagnostic message (default ``False``).
      jit: Whether to JIT-compile the optimization process (default ``True``).
        :term:`↪ More Details<jit>`
      sparse: Whether to sparsify the optimization process (default ``False``).
        :term:`↪ More Details<sparse>`
    """

    snapshop_update_freq: int
    learning_rate: Union[float, Callable] = 0.5

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchySVRGState:
        r"""The function initializes the optimizer state."""
        return SketchySVRGState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            full_grad=inexact_asarray(jnp.zeros((self.params_len,))),
            snapshot=params,
            step_size=inexact_asarray(0.0),
        )

    def _update_snapshot(
        self, params, state, data, reg, *args, **kwargs
    ) -> SketchySVRGState:
        _, full_grad = self._value_and_grad(params, *args, **kwargs, data=data, reg=reg)
        unraveled_full_grad, _ = ravel_tree(full_grad)
        return state._replace(
            full_grad=unraveled_full_grad,
            snapshot=params,
        )

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.grad_batch_size,), replace=False
        )

        # compute preconditioned variance-reduced stochastic gradient
        value, grad = self._value_and_grad(
            params,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )
        _, grad_snapshot = self._value_and_grad(
            state.snapshot,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )

        error = tree_l2_norm(grad, squared=False)

        unraveled_grad, unravel_fun = ravel_tree(grad)
        unraveled_grad_snapshot, _ = ravel_tree(grad_snapshot)

        direction = unravel_fun(
            self._grad_transform(
                unraveled_grad - unraveled_grad_snapshot + state.full_grad,
                state.precond,
            ),
        )

        # perform an update
        params = tree_add_scalar_mul(params, -state.step_size, direction)

        return SolverState(
            params=params,
            state=state._replace(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
            ),
        )

    def run(
        self, init_params: Any, data: ArrayLike, reg: float = 0.0, *args, **kwargs
    ) -> SolverState:
        r"""The function runs the optimization loop.

        Args:
          init_params: Initial value of the optimization variable.
          data: Full dataset. Expect an array of shape ``(num_samples, ...)``.
          reg: Regularization strength. Expect a non-negative value (default ``0``).
          *args: Additional positional arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
        Returns:
          Final optimization variable and solver state. The variable has the same shape
          as the provided initial value ``init_params``, and the state is an
          :class:`SketchySVRGState` object.
        """
        # initialize iterate and state
        params = init_params
        self.num_samples = jnp.shape(data)[0]
        self.params_len = jnp.size(ravel_tree(params)[0])
        state = self._init_state(params, data, reg, *args, **kwargs)

        # define partial functions
        update_params = lambda p, s: self._update_params(
            p, s, data, reg, *args, **kwargs
        )
        update_precond = lambda p, s: self._update_precond(
            p, s, data, reg, *args, **kwargs
        )
        update_snapshot = lambda p, s: self._update_snapshot(
            p, s, data, reg, *args, **kwargs
        )

        # sparsify functions if needed
        if self.sparse:
            update_params = sparse.sparsify(update_params)
            update_precond = sparse.sparsify(update_precond)
            update_snapshot = sparse.sparsify(update_snapshot)

        # JIT-compile functions if needed
        if self.jit:
            update_params = (
                jax.jit(update_params) if self.maxiter > 1 else update_params
            )
            update_precond = (
                jax.jit(update_precond)
                if (self.update_freq > 0 and self.maxiter / self.update_freq > 1)
                else update_precond
            )
            update_snapshot = (
                jax.jit(update_snapshot)
                if self.maxiter / self.snapshop_update_freq > 1
                else update_snapshot
            )

        # run the optimization loop
        for i in range(self.maxiter):
            # call custom pre-update function
            if self.pre_update:
                params, state = self.pre_update(
                    params, state, *args, **kwargs, data=data, reg=reg
                )

            # update snapshot
            if state.iter_num % self.snapshop_update_freq == 0:
                state = update_snapshot(params, state)

            # update preconditioner
            if (self.update_freq == 0 and state.iter_num == 0) or (
                self.update_freq > 0 and state.iter_num % self.update_freq == 0
            ):
                state = update_precond(params, state)

            # update iterate
            params, state = update_params(params, state)

            # break out of loop if tolerance has been reached
            if state.error < self.tol:
                print(
                    "Info: early termination because error tolerance has been reached."
                )
                break

        return SolverState(params=params, state=state)


class SketchySAGAState(NamedTuple):
    r"""The SketchySAGA optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      grad_table: Table of gradients of each individual component.
      table_avg: Average of the gradients in the table.
      step_size: Step size for the next update.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    grad_table: Array
    table_avg: Array
    step_size: Array


@dataclass(eq=False, kw_only=True)
class SketchySAGA(PromiseSolver):
    r"""The SketchySAGA optimizer.

    SketchySAGA is a preconditioned version of a minibatch variant [#f3]_ (b-nice SAGA)
    of SAGA [#f2]_. The optimizer maintains and updates a gradient table and table
    average. At each iteration, the optimizer computes auxiliary vector and uses it to
    update the variance-reduced stochastic gradient. The update is then based on the
    preconditioned variance-reduced stochastic gradient.

    .. note:: Because SketchySAGA computes gradient of each individual component
      function, the provided objective function ``fun`` or gradient function
      ``grad_fun`` needs to be compatible with 1-dimensional data input (*i.e.* when the
      data input is a vector representing a single sample). The following example
      expands the vector to a 2-dimensional array to handle the single sample case.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchySAGA

        def ridge_reg_objective(params, data, reg):
            # make data 2-dimensional if it is a vector of a single sample
            if jnp.ndim(data) == 1:
                data = jnp.expand_dims(data, axis=0)
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchySAGA(ridge_reg_objective, ...)
        opt.run(init_params, data, reg)

    .. rubric:: References

    .. [#f2] A\. Defazio, F. Bach, and S. Lacoste-Julien, `SAGA: A Fast Incremental
      Gradient Method With Support for Non-Strongly Convex Composite Objectives
      <https://papers.nips.cc/paper_files/paper/2014/hash/
      ede7e2b6d13a41ddf9f4bdef84fdc737-Abstract.html>`_, Advances in Neural Information
      Processing Systems, 27, 2014.
    .. [#f3] N\. Gazagnadou, R. M. Gower, and J. Salmon, `Optimal Mini-Batch and Step
      Sizes for SAGA <https://proceedings.mlr.press/v97/gazagnadou19a.html>`_, in
      *Proceedings of the 36*\ :sup:`th` *International Conference on Machine Learning*,
      Proceedings of Machine Learning Research (PMLR), 97: 2142-2150, 2019.

    - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned
      Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates
      <https://arxiv.org/abs/2309.02014>`_.

    Args:
      fun: Scalar-valued objective function compatible with 1-dimensional ``data``
        input. :term:`↪ More Details<fun>`
      grad_fun: Optional gradient oracle corresponding to the provided objective
        function ``fun``. It also needs to be compatible with 1-dimensional ``data``
        input. :term:`↪ More Details<grad_fun>`
      hvp_fun: Optional Hessian-vector product oracle for the Nyström subsampled Newton
        preconditioner. :term:`↪ More Details<hvp_fun>`
      sqrt_hess_fun: Required oracle that computes the square root of the Hessian matrix
        for the subsampled Newton preconditioner (*i.e.* cannot be empty if ``precond``
        is set to ``ssn``). :term:`↪ More Details<sqrt_hess_fun>`
      pre_update: Optional function to execute before optimizer's each update on the
        iterate. :term:`↪ More Details<pre_update>`
      precond: Type of preconditioner to use (default ``nyssn``). Either ``nyssn`` for
        Nyström subsampled Newton or ``ssn`` for subsampled Newton.
        :term:`↪ More Details<precond>`
      rho: Regularization parameter for the preconditioner. Expect a non-negative value
        (default ``1e-3``).
      rank: Rank of the Nyström subsampled Newton preconditioner. Expect a positive
        value (default ``10``).
      grad_batch_size: Size of the batch of data to compute stochastic gradient at each
        iteration. Expect a positive value.
      hess_batch_size: Size of the batch of data to estimate the stochastic Hessian when
        updating the preconditioner. Expect a positive value.
      update_freq: Update frequency of the preconditioner. When set to ``0`` or
        :math:`\infty` (*e.g.* ``jax.numpy.inf`` or ``numpy.inf``), the optimizer uses
        constant preconditioner that is constructed at the beginning of the optimization
        process.
      seed: Initial seed for the random number generator (default ``0``).
      learning_rate: Step size for applying updates (default ``0.5``). It can either be
        a fixed scalar value or a schedule (callable) based on step count.
        :term:`↪ More Details<learning_rate>`
      maxiter: Maximum number of iterations to run the optimizer (default ``20``).
        Expect a positive value.
      tol: Threshold of the gradient norm used for terminating the optimizer (default
        ``1e-3``).
      verbose: Whether to print diagnostic message (default ``False``).
      jit: Whether to JIT-compile the optimization process (default ``True``).
        :term:`↪ More Details<jit>`
      sparse: Whether to sparsify the optimization process (default ``False``).
        :term:`↪ More Details<sparse>`
    """

    learning_rate: Union[float, Callable] = 0.5

    def __post_init__(self):
        r"""The function overrides the superclass's method and constructs
        ``value_and_grad`` function that computes the gradient of each individual
        component."""
        if callable(self.grad_fun):
            comp_grad_fun = self.grad_fun
        else:
            comp_grad_fun = jax.grad(self.fun)

        def value_and_grad(params, data, reg, *args, **kwargs):
            g = lambda d: ravel_tree(
                comp_grad_fun(params, *args, **kwargs, data=d, reg=reg)
            )[0]
            grad_fun = jax.vmap(g, 0, 0)
            return self.fun(params, *args, **kwargs, data=data, reg=reg), grad_fun(data)

        self._value_and_grad = value_and_grad

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchySAGAState:
        r"""The function initializes the optimizer state."""
        return SketchySAGAState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            grad_table=inexact_asarray(jnp.zeros((self.num_samples, self.params_len))),
            table_avg=inexact_asarray(jnp.zeros((self.params_len,))),
            step_size=inexact_asarray(0.0),
        )

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.grad_batch_size,), replace=False
        )

        # compute stochastic gradients
        value, unraveled_grads = self._value_and_grad(
            params,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )
        error = jnp.linalg.norm(jnp.mean(unraveled_grads, axis=0))

        # compute auxiliary
        aux = jnp.sum(unraveled_grads - state.grad_table[batch_idx], axis=0)

        # compute preconditioned variance-reduced stochastic gradient
        _, unravel_fun = ravel_tree(params)
        direction = unravel_fun(
            self._grad_transform(
                state.table_avg + (1.0 / self.grad_batch_size) * aux, state.precond
            ),
        )

        # update table average and gradient table
        table_avg = state.table_avg + (1.0 / self.num_samples) * aux
        grad_table = state.grad_table.at[batch_idx].set(unraveled_grads)

        # perform an update
        params = tree_add_scalar_mul(params, -state.step_size, direction)

        return SolverState(
            params=params,
            state=state._replace(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
                grad_table=grad_table,
                table_avg=table_avg,
            ),
        )

    def run(
        self, init_params: Any, data: ArrayLike, reg: float = 0.0, *args, **kwargs
    ) -> SolverState:
        r"""The function runs the optimization loop.

        Args:
          init_params: Initial value of the optimization variable.
          data: Full dataset. Expect an array of shape ``(num_samples, ...)``.
          reg: Regularization strength. Expect a non-negative value (default ``0``).
          *args: Additional positional arguments to be passed to ``fun`` (and `
            `grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
        Returns:
          Final optimization variable and solver state. The variable has the same shape
          as the provided initial value ``init_params``, and the state is an
          :class:`SketchySAGAState` object.
        """
        # initialize iterate and state
        params = init_params
        self.num_samples = jnp.shape(data)[0]
        self.params_len = jnp.size(ravel_tree(params)[0])
        state = self._init_state(params, data, reg, *args, **kwargs)

        # define partial functions
        update_params = lambda p, s: self._update_params(
            p, s, data, reg, *args, **kwargs
        )
        update_precond = lambda p, s: self._update_precond(
            p, s, data, reg, *args, **kwargs
        )

        # sparsify functions if needed
        if self.sparse:
            update_params = sparse.sparsify(update_params)
            update_precond = sparse.sparsify(update_precond)

        # JIT-compile functions if needed
        if self.jit:
            update_params = (
                jax.jit(update_params) if self.maxiter > 1 else update_params
            )
            update_precond = (
                jax.jit(update_precond)
                if (self.update_freq > 0 and self.maxiter / self.update_freq > 1)
                else update_precond
            )

        # run the optimization loop
        for i in range(self.maxiter):
            # call custom pre-update function
            if self.pre_update:
                params, state = self.pre_update(
                    params, state, *args, **kwargs, data=data, reg=reg
                )

            # update preconditioner
            if (self.update_freq == 0 and state.iter_num == 0) or (
                self.update_freq > 0 and state.iter_num % self.update_freq == 0
            ):
                state = update_precond(params, state)

            # update iterate
            params, state = update_params(params, state)

            # break out of loop if tolerance has been reached
            if state.error < self.tol:
                print(
                    "Info: early termination because error tolerance has been reached."
                )
                break

        return SolverState(params=params, state=state)


class SketchyKatyushaState(NamedTuple):
    r"""The SketchyKatyusha optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      full_grad: Full gradient at the snapshot.
      snapshot: Snapshot of the iterate.
      z: Momentum of the iterate.
      step_size: Step size for the next update.
      L: Smoothness constant estimate.
      sigma: Inverse condition number estimate.
      theta: Momentum parameter.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    full_grad: Array
    snapshot: Array
    z: Array
    step_size: Array
    L: Array
    sigma: Array
    theta: Array


@dataclass(eq=False, kw_only=True)
class SketchyKatyusha(PromiseSolver):
    r"""The SketchyKatyusha optimizer.

    SketchyKatyusha is a preconditioned version of Loopless Katyusha [#f5]_ that extends
    the original Katyusha [#f4]_. The optimizer calculates the preconditioned
    variance-reduced stochastic gradient and performs momentum update. The optimizer
    periodically updates the preconditioner, and probabilistically updates snapshot and
    full gradient.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchyKatyusha

        def ridge_reg_objective(params, data, reg):
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchyKatyusha(ridge_reg_objective, ...)
        opt.run(init_params, data, reg)

    .. rubric:: References

    .. [#f4] Z\. Allen-Zhu, `Katyusha: The First Direct Acceleration of Stochastic
      Gradient Methods <https://jmlr.org/papers/v18/16-410.html>`_, Journal of Machine
      Learning Research, 18(221): 1-51, 2018.
    .. [#f5] D\. Kovalev, S. Horváth, and P. Richtárik, `Don't Jump Through Hoops and
      Remove Those Loops: SVRG and Katyusha are Better Without the Outer Loop
      <https://proceedings.mlr.press/v117/kovalev20a.html>`_, in *Proceedings of the
      31*\ :sup:`th` *International Conference on Algorithmic Learning Theory*,
      Proceedings of Machine Learning Research (PMLR), 117: 451-467, 2020.

    - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned
      Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates
      <https://arxiv.org/abs/2309.02014>`_.

    Args:
      fun: Scalar-valued objective function. :term:`↪ More Details<fun>`
      grad_fun: Optional gradient oracle corresponding to the provided objective
        function ``fun``. :term:`↪ More Details<grad_fun>`
      hvp_fun: Optional Hessian-vector product oracle for the Nyström subsampled Newton
        preconditioner. :term:`↪ More Details<hvp_fun>`
      sqrt_hess_fun: Required oracle that computes the square root of the Hessian matrix
        for the subsampled Newton preconditioner (*i.e.* cannot be empty if ``precond``
        is set to ``ssn``). :term:`↪ More Details<sqrt_hess_fun>`
      pre_update: Optional function to execute before optimizer's each update on the
        iterate. :term:`↪ More Details<pre_update>`
      precond: Type of preconditioner to use (default ``nyssn``). Either ``nyssn`` for
        Nyström subsampled Newton or ``ssn`` for subsampled Newton.
        :term:`↪ More Details<precond>`
      rho: Regularization parameter for the preconditioner. Expect a non-negative value
        (default ``1e-3``).
      rank: Rank of the Nyström subsampled Newton preconditioner. Expect a positive
        value (default ``10``).
      mu: Strong convexity parameter. Expect a positive value.
      grad_batch_size: Size of the batch of data to compute stochastic gradient at each
        iteration. Expect a positive value.
      hess_batch_size: Size of the batch of data to estimate the stochastic Hessian when
        updating the preconditioner. Expect a positive value.
      update_freq: Update frequency of the preconditioner. When set to ``0`` or
        :math:`\infty` (*e.g.* ``jax.numpy.inf`` or ``numpy.inf``), the optimizer uses
        constant preconditioner that is constructed at the beginning of the optimization
        process.
      snapshop_update_prob: Probability of updating the snapshot. Expect a value in
        :math:`(0,1)`.
      seed: Initial seed for the random number generator (default ``0``).
      momentum_param: Momentum parameter (default ``1/2``).
      momentum_multiplier: Momentum multiplier (default ``2/3``).
      maxiter: Maximum number of iterations to run the optimizer (default ``20``).
        Expect a positive value.
      tol: Threshold of the gradient norm used for terminating the optimizer (default
        ``1e-3``).
      verbose: Whether to print diagnostic message (default ``False``).
      jit: Whether to JIT-compile the optimization process (default ``True``).
        :term:`↪ More Details<jit>`
      sparse: Whether to sparsify the optimization process (default ``False``).
        :term:`↪ More Details<sparse>`
    """

    mu: float
    momentum_param: float = 1.0 / 2
    momentum_multiplier: float = 2.0 / 3
    snapshop_update_prob: float

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchyKatyushaState:
        r"""The function initializes the optimizer state."""
        _, full_grad = self._value_and_grad(params, *args, **kwargs, data=data, reg=reg)
        full_grad, _ = ravel_tree(full_grad)
        return SketchyKatyushaState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            full_grad=full_grad,
            snapshot=params,
            z=params,
            step_size=inexact_asarray(0.0),
            L=inexact_asarray(0.0),
            sigma=inexact_asarray(0.0),
            theta=inexact_asarray(0.0),
        )

    def _update_step_size(self, labda, state):
        r"""The function overrides the superclass's method and updates relevant values
        using the provided preconditioned smoothness constant labda."""
        sigma = self.mu / labda
        theta = jnp.minimum(
            0.5, jnp.sqrt(self.momentum_multiplier * sigma * self.num_samples)
        )
        step_size = self.momentum_param / (theta * (1 + self.momentum_param))
        return state._replace(
            step_size=step_size,
            L=labda,
            sigma=sigma,
            theta=theta,
        )

    def _update_snapshot(self, u, params, state, data, reg, *args, **kwargs):
        r"""The function updates snapshot and full gradient with specified update
        probability."""

        def update():
            _, full_grad = self._value_and_grad(
                params, *args, **kwargs, data=data, reg=reg
            )
            full_grad, _ = ravel_tree(full_grad)
            return state._replace(full_grad=full_grad, snapshot=params)

        return jax.lax.cond(u <= self.snapshop_update_prob, update, lambda: state)

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey1, subkey2 = jax.random.split(state.key, num=3)
        batch_idx = jax.random.choice(
            subkey1, self.num_samples, (self.grad_batch_size,), replace=False
        )
        u = jax.random.uniform(subkey2)

        # compute negative momentum
        x = tree_scalar_mul(state.theta, state.z)
        x = tree_add_scalar_mul(x, self.momentum_param, state.snapshot)
        x = tree_add_scalar_mul(x, 1 - state.theta - self.momentum_param, params)

        # compute preconditioned variance-reduced accelerated stochastic gradient
        value, grad = self._value_and_grad(
            x, *args, **kwargs, data=data[batch_idx], reg=reg
        )
        _, grad_snapshot = self._value_and_grad(
            state.snapshot,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )

        error = tree_l2_norm(grad, squared=False)

        unraveled_grad, unravel_fun = ravel_tree(grad)
        unraveled_grad_snapshot, _ = ravel_tree(grad_snapshot)

        direction = unravel_fun(
            self._grad_transform(
                unraveled_grad - unraveled_grad_snapshot + state.full_grad,
                state.precond,
            ),
        )

        # update snapshot
        state = self._update_snapshot(u, params, state, data, reg, *args, **kwargs)

        # perform an update
        z = tree_scalar_mul(state.step_size * state.sigma, x)
        z = tree_add(z, state.z)
        z = tree_add_scalar_mul(z, -state.step_size / state.L, direction)
        z = tree_scalar_mul(1.0 / (1 + state.step_size * state.sigma), z)
        params = tree_add_scalar_mul(x, state.theta, tree_sub(z, state.z))

        return SolverState(
            params=params,
            state=state._replace(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
                z=z,
            ),
        )

    def run(
        self, init_params: Any, data: ArrayLike, reg: float = 0.0, *args, **kwargs
    ) -> SolverState:
        r"""The function runs the optimization loop.

        Args:
          init_params: Initial value of the optimization variable.
          data: Full dataset. Expect an array of shape ``(num_samples, ...)``.
          reg: Regularization strength. Expect a non-negative value (default ``0``).
          *args: Additional positional arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
        Returns:
          Final optimization variable and solver state. The variable has the same shape
          as the provided initial value ``init_params``, and the state is an
          :class:`SketchyKatyushaState` object.
        """
        # initialize iterate and state
        params = init_params
        self.num_samples = jnp.shape(data)[0]
        self.params_len = jnp.size(ravel_tree(params)[0])
        state = self._init_state(params, data, reg, *args, **kwargs)

        # define partial functions
        update_params = lambda p, s: self._update_params(
            p, s, data, reg, *args, **kwargs
        )
        update_precond = lambda p, s: self._update_precond(
            p, s, data, reg, *args, **kwargs
        )

        # sparsify functions if needed
        if self.sparse:
            update_params = sparse.sparsify(update_params)
            update_precond = sparse.sparsify(update_precond)

        # JIT-compile functions if needed
        if self.jit:
            update_params = (
                jax.jit(update_params) if self.maxiter > 1 else update_params
            )
            update_precond = (
                jax.jit(update_precond)
                if (self.update_freq > 0 and self.maxiter / self.update_freq > 1)
                else update_precond
            )

        # run the optimization loop
        for i in range(self.maxiter):
            # call custom pre-update function
            if self.pre_update:
                params, state = self.pre_update(
                    params, state, *args, **kwargs, data=data, reg=reg
                )

            # update preconditioner
            if (self.update_freq == 0 and state.iter_num == 0) or (
                self.update_freq > 0 and state.iter_num % self.update_freq == 0
            ):
                state = update_precond(params, state)

            # update iterate
            params, state = update_params(params, state)

            # break out of loop if tolerance has been reached
            if state.error < self.tol:
                print(
                    "Info: early termination because error tolerance has been reached."
                )
                break

        return SolverState(params=params, state=state)
