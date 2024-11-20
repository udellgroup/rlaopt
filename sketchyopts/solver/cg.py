import jax

from sketchyopts.util import tree_add_scalar_mul, tree_l2_norm, tree_sub, tree_vdot


def abstract_cg(A, b, mu, x0, tol, maxiter, M):
    r"""The preconditioned conjugate gradient method.

    The function solves the regularized linear system :math:`(A + \mu I) x = b` using
    the preconditioned conjugate gradient method.

    Args:
      A: An operator representing a positive-semidefinite matrix.
      b: A vector or pytree giving the righthand side of the regularized linear system.
      mu: Damping parameter. Expect a non-negative value.
      x0: Initial guess for the solution (same size as righthand side ``b``).
      tol: Solution tolerance.
      maxiter: Maximum number of iterations.
      M: Preconditioner for the regularized linear system.

    Returns:
      Approximate solution to the regualrized linear system. Solution has the same size
      as righthand side ``b``.
    """

    # matrix-vector product for regularized linear operator, i.e. compute (A + mu I) x
    @jax.jit
    def regularized_A(x):
        return tree_add_scalar_mul(A @ x, mu, x)

    # condition evaluation: check if the maximum iterations are reached or the
    # residual is below tolerance
    def cond_fun(value):
        _, r, _, _, k = value
        return (k < maxiter) & (tree_l2_norm(r) > tol)

    # perform the PCG iteration step
    def body_fun(value):
        x, r, z, p, k = value
        v = regularized_A(p)
        gamma = tree_vdot(r, z)
        alpha = gamma / tree_vdot(p, v)
        x_ = tree_add_scalar_mul(x, alpha, p)
        r_ = tree_add_scalar_mul(r, -alpha, v)
        z_ = M(r_)
        beta = tree_vdot(r_, z_) / gamma
        p_ = tree_add_scalar_mul(z_, beta, p)
        return x_, r_, z_, p_, k + 1

    # initial step
    r0 = tree_sub(b, regularized_A(x0))
    p0 = z0 = M(r0)
    initial_value = (x0, r0, z0, p0, 0)

    # perform iterative solve
    x_final, r_final, z_final, p_final, k_final = jax.lax.while_loop(
        cond_fun, body_fun, initial_value
    )

    return x_final, r_final, z_final, p_final, k_final
