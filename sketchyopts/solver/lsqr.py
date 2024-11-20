import jax
import jax.numpy as jnp

from sketchyopts.util import (
    default_floating_dtype,
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_scalar_mul,
    tree_sub,
)


def abstract_lsqr(A, b, x0, tol, maxiter, M, M_transpose):
    r"""Solve the linear least-squares problem using the LSQR algorithm.

    The function solves the linear least-squares problem using the preconditioned LSQR
    algorithm. The algorithm terminates if one of the following conditions is met:

    - The number of iterations reaches the maximum number of iterations.
    - The norm of the residual :math:`\lVert (AM)x - b \rVert` is within the specified
      tolerance :math:`\text{tol}\big(\lVert AM \rVert \cdot \lVert x \rVert +
      \lVert b \rVert \big)`.
    - The norm of the quantity :math:`\lVert (AM)^{\mathsf{T}} (b - AMx) \rVert` is
      within the specified tolerance :math:`\text{tol} \cdot \lVert AM \rVert \cdot
      \lVert b - AMx \rVert`.
    - an estimate of the condition number of :math:`AM` exceeds :math:`10^8`.

    Args:
      A: An operator representing a matrix.
      b: A vector or pytree representing the righthand side of the linear least-squares
        problem.
      x0: Initial guess for the solution (same size as righthand side ``b``).
      tol: Solution tolerance.
      maxiter: Maximum number of iterations.
      M: Preconditioner for the linear least-squares.
      M_transpose: Transpose of the preconditioner for the linear least-squares.

    Returns:
      Approximate solution to the original linear least-squares problem (without right
        preconditioner), norm of the residual, estimate of the norm of the
        preconditioned operator :math:`AM`, and the number of iterations.
    """

    bnorm = tree_l2_norm(b)
    ctol = 1.0 / 1e8
    eps = jnp.finfo(default_floating_dtype).eps
    atol, btol = tol, tol

    def cond_fun(value):
        (
            _,
            _,
            _,
            _,
            _,
            alpha,
            _,
            anorm,
            ddnorm,
            xnorm,
            _,
            _,
            _,
            _,
            phibar,
            tau,
            num_iter,
        ) = value

        acond = anorm * jnp.sqrt(ddnorm)
        rnorm = phibar
        arnorm = alpha * jnp.abs(tau)

        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1.0 / (acond + eps)
        rtol = btol + atol * anorm * xnorm / bnorm

        return (num_iter < maxiter) & (test1 > rtol) & (test2 > atol) & (test3 > ctol)

    def body_fun(value):
        (
            x,
            u,
            v,
            w,
            z,
            alpha,
            beta,
            anorm,
            ddnorm,
            xnorm,
            xxnorm,
            cs2,
            sn2,
            rhobar,
            phibar,
            tau,
            num_iter,
        ) = value

        # bidiagonalization
        u = tree_add_scalar_mul(A.mv(M(v)), -alpha, u)
        beta = tree_l2_norm(u)
        u = tree_scalar_mul(1.0 / beta, u)
        anorm = jnp.sqrt(anorm**2 + alpha**2 + beta**2)
        v = tree_add_scalar_mul(M_transpose(A.transpose().mv(u)), -beta, v)
        alpha = tree_l2_norm(v)
        v = tree_scalar_mul(1.0 / alpha, v)

        # apply orthogonal transformation
        rho = jnp.sqrt(rhobar**2 + beta**2)
        cs = rhobar / rho
        sn = beta / rho
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # update x and w
        t1 = phi / rho
        t2 = -theta / rho
        dk = tree_scalar_mul(1.0 / rho, w)
        x = tree_add_scalar_mul(x, t1, w)
        w = tree_add_scalar_mul(v, t2, w)
        ddnorm = ddnorm + tree_l2_norm(dk, True)

        # estimate norms for convergence tests
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = jnp.sqrt(xxnorm + zbar**2)
        gamma = jnp.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        return (
            x,
            u,
            v,
            w,
            z,
            alpha,
            beta,
            anorm,
            ddnorm,
            xnorm,
            xxnorm,
            cs2,
            sn2,
            rhobar,
            phibar,
            tau,
            num_iter + 1,
        )

    # initialization
    u = tree_sub(b, A.mv(M(x0)))
    beta = tree_l2_norm(u)
    if beta > 0:
        u = tree_scalar_mul(1.0 / beta, u)
        v = M_transpose(A.transpose().mv(u))
        alpha = tree_l2_norm(v)
    else:
        v = x0.copy()
        alpha = 0.0

    if alpha > 0:
        v = tree_scalar_mul(1.0 / alpha, v)

    w = v.copy()
    z = 0.0
    anorm = 0.0
    ddnorm = 0.0
    xnorm = 0
    xxnorm = 0
    cs2 = -1.0
    sn2 = 0.0
    rhobar = alpha
    phibar = beta
    tau = 0.0
    num_iter = 0

    init_value = (
        x0,
        u,
        v,
        w,
        z,
        alpha,
        beta,
        anorm,
        ddnorm,
        xnorm,
        xxnorm,
        cs2,
        sn2,
        rhobar,
        phibar,
        tau,
        num_iter,
    )

    # perform iterative solve
    (
        x_final,
        _,
        _,
        _,
        _,
        _,
        _,
        anorm_final,
        ddnorm_final,
        _,
        _,
        _,
        _,
        _,
        phibar_final,
        _,
        iter_final,
    ) = jax.lax.while_loop(cond_fun, body_fun, init_value)

    return M(x_final), phibar_final, anorm_final * jnp.sqrt(ddnorm_final), iter_final
