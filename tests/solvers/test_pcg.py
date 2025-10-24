"""Tests for the PCG solver."""

import pytest
import torch

from rlaopt.linalg import IdentityConfig, LinSys, NystromConfig
from rlaopt.solvers.pcg import PCG, PCGConfig, PCGStoppingCriteria

TOLERANCES = {torch.float32: 1e-4, torch.float64: 1e-10}
MAX_ITERS = 100


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    """Torch dtype for the test."""
    return request.param


@pytest.fixture
def tol(precision):
    """Convergence tolerance based on precision."""
    return TOLERANCES[precision]


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
    """Generate a positive-definite linear system.

    Creates a system of the form (A + reg*I)W = B where A is symmetric
    positive-definite.

    Args:
        n: Size of the matrix A (n x n)
        num_rhs: Number of right-hand side vectors
        reg: Regularization parameter
        precision: Torch dtype for tensors
        seed: Random seed for reproducibility

    Returns:
        LinSys object representing the linear system
    """
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

    def test_identity_preconditioner(self, reset_torch_state, precision, tol):
        """Test PCG with identity preconditioner."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=100, num_rhs=5, reg=1e-3, precision=precision
            )
            preconditioner_config = IdentityConfig()
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)

    def test_nystrom_preconditioner(self, reset_torch_state, precision, tol):
        """Test PCG with Nystrom preconditioner."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=100, num_rhs=5, reg=1e-3, precision=precision
            )
            preconditioner_config = NystromConfig(rank=20, base_damping=1e-3)
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)

    def test_single_rhs(self, reset_torch_state, precision, tol):
        """Test PCG with a single right-hand side vector."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=50, num_rhs=1, reg=1e-3, precision=precision
            )
            preconditioner_config = IdentityConfig()
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)

    def test_large_system(self, reset_torch_state, precision, tol):
        """Test PCG on a larger system with Nystrom preconditioner."""

        def setup_problem():
            lin_sys = generate_positive_definite_system(
                n=500, num_rhs=10, reg=1e-3, precision=precision
            )
            preconditioner_config = NystromConfig(rank=50, base_damping=1e-3)
            return lin_sys, preconditioner_config

        _test_pcg_solver(precision, tol, setup_problem)


# ============================================================================
# Helper Functions
# ============================================================================


def _test_pcg_solver(precision, tol, setup_fn):
    """Common test structure for PCG solver.

    Args:
        precision: Torch dtype for the test
        tol: Convergence tolerance
        setup_fn: Function that returns (lin_sys, preconditioner_config)
    """
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

    # Verify solution is actually correct
    actual_rel_res_norm = lin_sys.compute_residual_norm(params, relative=True)
    assert (actual_rel_res_norm <= tol).all(), (
        f"Solution verification failed: "
        f"max relative residual norm {actual_rel_res_norm.max().item()} > "
        f"tolerance {tol}"
    )

    # Test solve method
    params, _ = solver.solve(stopping_criteria=stopping_criteria)

    # Verify solution is actually correct
    actual_rel_res_norm = lin_sys.compute_residual_norm(params, relative=True)
    assert (actual_rel_res_norm <= tol).all(), (
        f"Solution verification failed: "
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
