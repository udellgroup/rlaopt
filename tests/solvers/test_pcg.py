"""Tests for the PCG solver."""

import pytest
import torch

from rlaopt.linalg import IdentityConfig, LinSys, NystromConfig
from rlaopt.solvers.pcg import PCG, PCGConfig, PCGStoppingCriteria

TOLERANCES = {torch.float32: 1e-4, torch.float64: 1e-10}
MAX_ITERS = 100
GRAD_TEST_RTOL = {torch.float32: 1e-2, torch.float64: 1e-4}


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    """Torch dtype for the test."""
    return request.param


@pytest.fixture
def tol(precision):
    """Convergence tolerance based on precision."""
    return TOLERANCES[precision]


@pytest.fixture
def reg():
    """Regularization parameter for linear systems."""
    return 1e-3


@pytest.fixture
def reset_torch_state():
    """Fixture to reset torch default dtype after each test."""
    original_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(original_dtype)


# ============================================================================
# Data Generation Helpers
# ============================================================================


def generate_positive_definite_system(
    n=100, num_rhs=10, reg=1e-3, precision=torch.float32, seed=0
):
    """Generate a positive-definite linear system."""
    torch.manual_seed(seed)

    # Create eigenvalues that decay polynomially (ill-conditioned)
    eigvals = torch.arange(1, n + 1, dtype=precision) ** -2.0

    # Create random orthogonal matrix
    U = torch.randn(n, n, dtype=precision)
    U = torch.linalg.qr(U).Q

    # Construct A = U * diag(eigvals) * U^T
    A = U @ torch.diag(eigvals) @ U.T

    # Create random right-hand side
    B = torch.randn(n, num_rhs, dtype=precision)

    return LinSys(A, B, reg=reg)


# ============================================================================
# Test Class
# ============================================================================


class TestPCG:
    """Tests for the PCG solver."""

    def test_identity_preconditioner(self, reset_torch_state, precision, tol, reg):
        """Test PCG with identity preconditioner."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=100, num_rhs=5, reg=reg, precision=precision
            )
            preconditioner_config = IdentityConfig()
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)

    def test_nystrom_preconditioner(self, reset_torch_state, precision, tol, reg):
        """Test PCG with Nystrom preconditioner."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=100, num_rhs=5, reg=reg, precision=precision
            )
            preconditioner_config = NystromConfig(rank_init=20, base_damping=reg)
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)

    def test_single_rhs(self, reset_torch_state, precision, tol, reg):
        """Test PCG with a single right-hand side vector."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=50, num_rhs=1, reg=reg, precision=precision
            )
            preconditioner_config = IdentityConfig()
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)

    def test_large_system(self, reset_torch_state, precision, tol, reg):
        """Test PCG on a larger system with Nystrom preconditioner."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=500, num_rhs=10, reg=reg, precision=precision
            )
            preconditioner_config = NystromConfig(rank_init=50, base_damping=reg)
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)


# ============================================================================
# Helper Functions
# ============================================================================


def _test_pcg_solver(precision, tol, setup_fn):
    """Common test structure for PCG solver."""
    torch.set_default_dtype(precision)
    lin_sys, preconditioner_config = setup_fn()
    _solve_and_verify(lin_sys, preconditioner_config, tol)


def _solve_and_verify(lin_sys, preconditioner_config, tol):
    """Test that PCG solver works correctly.

    Tests both step-by-step iteration and the solve method.
    """
    config = PCGConfig(preconditioner_config=preconditioner_config)
    solver = PCG(lin_sys, config)
    stopping_criteria = PCGStoppingCriteria(tol=tol, max_iters=MAX_ITERS)

    # Test step-by-step solving
    params = lin_sys.w.clone()
    state = solver.init_state(params)
    params, _ = _loop(params, state, solver, stopping_criteria)
    _verify_solution(lin_sys, params, tol, "Step-by-step solving")

    # Test solve method
    params, _ = solver.solve(stopping_criteria=stopping_criteria)
    _verify_solution(lin_sys, params, tol, "Solve method")


def _verify_solution(lin_sys, params, tol, method_name):
    """Verify that the solution satisfies the tolerance."""
    actual_rel_res_norm = lin_sys.compute_residual_norm(params, relative=True)
    assert (actual_rel_res_norm <= tol).all(), (
        f"{method_name} failed: "
        f"max relative residual norm {actual_rel_res_norm.max().item()} > "
        f"tolerance {tol}"
    )


def _loop(params, state, solver, stopping_criteria):
    """Run PCG solver loop until convergence or max iterations."""
    for _ in range(stopping_criteria.max_iters):
        params, state = solver.step(params, state)
        if (state.res_norm <= stopping_criteria.tol).all():
            break
    return params, state.res_norm


# ============================================================================
# Differentiation Tests
# ============================================================================


class TestPCGDifferentiation:
    """Tests for gradient computation through PCG solver."""

    @pytest.mark.parametrize("wrt", ["B", "reg"])
    def test_gradients(self, reset_torch_state, precision, wrt):
        """Test gradient of solution norm squared w.r.t. B or reg."""
        torch.set_default_dtype(precision)
        _test_gradient(wrt, precision)


# ============================================================================
# Differentiation Helper Functions
# ============================================================================


def _solve_system(A, B, reg):
    """Solve (A + reg*I)w = B."""
    A_reg = A + reg * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A_reg, B)


def _solved_params_sq_norm(A, B, reg, preconditioner_config, tol):
    """Compute ||w||^2 where w solves (A + reg*I)w = B using PCG."""
    lin_sys = LinSys(A, B, reg)
    solver = PCG(lin_sys, PCGConfig(preconditioner_config=preconditioner_config))
    params, _ = solver.solve(
        stopping_criteria=PCGStoppingCriteria(max_iters=MAX_ITERS, tol=tol)
    )
    return torch.linalg.norm(params) ** 2


def _analytical_gradient(A, B, reg, wrt):
    """Compute analytical gradient of ||w||^2 w.r.t. specified parameter."""
    w = _solve_system(A, B, reg)
    inv_w = _solve_system(A, w, reg)  # (A + reg*I)^{-1} @ w

    if wrt == "B":
        # Gradient w.r.t. B: 2 * (A + reg*I)^{-1} @ w
        return 2 * inv_w
    elif wrt == "reg":
        # Gradient w.r.t. reg: -2 * w^T @ (A + reg*I)^{-1} @ w
        return -2 * torch.sum(w * inv_w)
    else:
        raise ValueError(f"Unknown parameter: {wrt}")


def _test_gradient(wrt, precision):
    """Test gradient computation w.r.t. specified parameter (B or reg)."""
    n = 100
    torch.manual_seed(42)

    # Generate test problem
    eigvals = torch.arange(1, n + 1, dtype=precision) ** -2.0
    U = torch.randn(n, n, dtype=precision)
    U = torch.linalg.qr(U).Q
    A = U @ torch.diag(eigvals) @ U.T
    B = torch.randn(n, dtype=precision) / (n**0.5)
    reg = torch.tensor(1e-3, dtype=precision)

    tol = TOLERANCES[precision]
    preconditioner_config = NystromConfig(rank_init=50, base_damping=reg.item())

    # Map parameter name to argnums for torch.func.grad
    argnums_map = {"B": 0, "reg": 1}

    # Compute gradient via autograd
    grad_autograd = torch.func.grad(
        lambda B_var, reg_var: _solved_params_sq_norm(
            A, B_var, reg_var, preconditioner_config, tol
        ),
        argnums=argnums_map[wrt],
    )(B, reg)

    # Compute analytical gradient
    grad_analytical = _analytical_gradient(A, B, reg, wrt)

    # Compare
    rtol = GRAD_TEST_RTOL[precision]

    rel_error = torch.abs(
        (grad_autograd - grad_analytical) / (torch.abs(grad_analytical) + 1e-8)
    )
    max_rel_error = torch.max(rel_error) if grad_analytical.ndim > 0 else rel_error

    assert torch.allclose(grad_autograd, grad_analytical, rtol=rtol), (
        f"Gradient w.r.t. {wrt} mismatch: max relative error = {max_rel_error}"
    )
