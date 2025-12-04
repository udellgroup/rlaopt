"""Tests for the ADMM solver."""

import pytest
import torch

from rlaopt.atoms import Box, L1Norm, NonNegative, SumSquares
from rlaopt.expression import Variable
from rlaopt.solvers import ConvergenceStatus
from rlaopt.solvers.admm import ADMM, ADMMConfig, ADMMStoppingCriteria

TOLERANCES = {torch.float32: 1e-3, torch.float64: 1e-6}
MAX_ITERS = 1000


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


def generate_least_squares_data(n=256, p=128, precision=torch.float32, seed=0):
    """Generate random data for least squares problems."""
    torch.manual_seed(seed)
    A = torch.randn(n, p, dtype=precision) / (n**0.5)
    b = torch.randn(n, dtype=precision) / (n**0.5)
    x = Variable(torch.zeros(p, dtype=precision))
    return A, b, x


def generate_lasso_data(n=512, p=64, s=16, precision=torch.float32, seed=0):
    """Generate random data for LASSO problems with sparse ground truth."""
    torch.manual_seed(seed)

    # Generate sparse ground truth
    J = torch.randperm(p)[:s]
    x_star = torch.zeros(p, dtype=precision)
    x_star[J] = torch.randn(s, dtype=precision) / (s**0.5)

    # Generate measurement matrix and noisy observations
    A = torch.randn(n, p, dtype=precision) / (n**0.5)
    b = A @ x_star + 0.01 * torch.randn(n, dtype=precision)

    x = Variable(torch.zeros(p, dtype=precision))
    return A, b, x, x_star


# ============================================================================
# Test Class: Simple Objectives (from test_prox_grad.py)
# ============================================================================


class TestADMMSimpleObjectives:
    """Tests for ADMM solver on simple objectives similar to ProxGrad tests."""

    def test_box_constrained(self, reset_torch_state, precision, tol):
        """Test ADMM on box-constrained least squares problem."""
        torch.set_default_dtype(precision)
        A, b, x = generate_least_squares_data(n=256, p=128, precision=precision)
        lower = -torch.tensor(2.0)
        upper = torch.tensor(1.0)
        obj = SumSquares(A @ x - b) + Box(x, lower=lower, upper=upper)
        _solve_and_verify(obj, tol)

    def test_nonnegative_constrained(self, reset_torch_state, precision, tol):
        """Test ADMM on nonnegative-constrained least squares problem."""
        torch.set_default_dtype(precision)
        A, b, x = generate_least_squares_data(n=256, p=128, precision=precision)
        obj = SumSquares(A @ x - b) + NonNegative(x)
        _solve_and_verify(obj, tol)

    def test_lasso(self, reset_torch_state, precision, tol):
        """Test ADMM on LASSO problem."""
        torch.set_default_dtype(precision)
        A, b, x, _ = generate_lasso_data(n=512, p=64, s=16, precision=precision)
        mu = 0.1 * torch.linalg.norm(A.T @ b, ord=torch.inf)
        obj = SumSquares(A @ x - b) + L1Norm(x, scaling=mu)
        _solve_and_verify(obj, tol)

    def test_elastic_net(self, reset_torch_state, precision, tol):
        """Test ADMM on elastic net problem (L2 + L1 regularization)."""
        torch.set_default_dtype(precision)
        A, b, x, _ = generate_lasso_data(n=512, p=64, s=16, precision=precision)
        mu = 0.1 * torch.linalg.norm(A.T @ b, ord=torch.inf)
        alpha = 0.01
        obj = SumSquares(A @ x - b) + alpha * SumSquares(x) + L1Norm(x, scaling=mu)
        _solve_and_verify(obj, tol)

    def test_pure_nonsmooth(self, reset_torch_state, precision, tol):
        """Test ADMM on pure non-smooth objective."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        p = 64
        x = Variable(torch.randn(p, dtype=precision), name="x")
        obj = L1Norm(x)
        _solve_and_verify(obj, tol)


# ============================================================================
# Test Class: Complex Objectives (inspired by test_admm_split.py)
# ============================================================================


class TestADMMComplexObjectives:
    """Tests for ADMM solver on complex objectives with multiple variables."""

    def test_nonsmooth_subset_of_smooth(self, reset_torch_state, precision, tol):
        """Test when non-smooth variables are subset of smooth variables."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 128, 32
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        y = Variable(torch.zeros(16, dtype=precision), name="y")
        lambd = 0.1
        obj = SumSquares(A @ x - b) + SumSquares(y) + L1Norm(x, scaling=lambd)
        _solve_and_verify(obj, tol)

    def test_nonsmooth_disjoint_from_smooth(self, reset_torch_state, precision, tol):
        """Test non-smooth variables disjoint from smooth variables."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 128, 32
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        z = Variable(torch.zeros(16, dtype=precision), name="z")
        obj = SumSquares(A @ x - b) + L1Norm(z, scaling=0.1)
        _solve_and_verify(obj, tol)

    def test_complex_mixed_variables(self, reset_torch_state, precision, tol):
        """Test complex case with multiple smooth and non-smooth terms."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 64, 16
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        y = Variable(torch.zeros(p, dtype=precision), name="y")
        z = Variable(torch.zeros(p // 2, dtype=precision), name="z")
        lambd = 0.05
        obj = (
            SumSquares(A @ x - b)
            + SumSquares(A @ y - b)
            + SumSquares(x + y)
            + L1Norm(x, scaling=lambd)
            + L1Norm(y, scaling=lambd)
            + L1Norm(z, scaling=lambd)
        )
        _solve_and_verify(obj, tol)

    def test_multiple_regularizers_same_variable(
        self, reset_torch_state, precision, tol
    ):
        """Test multiple non-smooth regularizers on the same variable."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 128, 32
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        lambd = 0.05
        obj = SumSquares(A @ x - b) + L1Norm(x, scaling=lambd) + NonNegative(x)
        _solve_and_verify(obj, tol)

    def test_affine_transformed_regularizer(self, reset_torch_state, precision, tol):
        """Test non-smooth regularizer on affine transformation of variables."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 128, 32
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        C = torch.randn(16, p, dtype=precision) / (p**0.5)
        d = torch.randn(16, dtype=precision)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        lambd = 0.1
        obj = SumSquares(A @ x - b) + L1Norm(C @ x - d, scaling=lambd)
        _solve_and_verify(obj, tol)

    def test_multi_variable_affine_regularizer(self, reset_torch_state, precision, tol):
        """Test non-smooth regularizer involving affine combination of variables."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 64, 16
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        y = Variable(torch.zeros(p // 2, dtype=precision), name="y")
        C = torch.randn(p // 2, p, dtype=precision) / (p**0.5)
        lambd = 0.05

        obj = SumSquares(A @ x - b) + SumSquares(y) + L1Norm(C @ x + y, scaling=lambd)
        _solve_and_verify(obj, tol)


# ============================================================================
# Test Class: Error Conditions
# ============================================================================


class TestADMMErrors:
    """Tests for error conditions in ADMM initialization."""

    def test_invalid_objective_type(self):
        """Test that ADMM raises ValueError for non-Expression objective."""
        config = ADMMConfig()
        with pytest.raises(ValueError, match="Expression objective"):
            ADMM("not an expression", config)

    def test_invalid_config_type(self):
        """Test that ADMM raises ValueError for non-ADMMConfig."""
        x = Variable(torch.ones(5), name="x")
        obj = SumSquares(x) + L1Norm(x)
        with pytest.raises(ValueError, match="ADMMConfig configuration"):
            ADMM(obj, "not a config")

    def test_all_smooth_objective(self):
        """Test that ADMM raises error for all-smooth objective."""
        x = Variable(torch.ones(5), name="x")
        obj = SumSquares(x)
        config = ADMMConfig()
        # ADMMSplit should raise error for all-smooth objective
        with pytest.raises(ValueError, match="at least one non-smooth term"):
            ADMM(obj, config)


# ============================================================================
# Test Class: Different Precisions and Configurations
# ============================================================================


class TestADMMConfigurations:
    """Tests for different ADMM configurations."""

    def test_different_rho_values(self, reset_torch_state, precision, tol):
        """Test ADMM with different initial rho values."""
        torch.set_default_dtype(precision)
        A, b, x = generate_least_squares_data(n=128, p=64, precision=precision)
        obj = SumSquares(A @ x - b) + L1Norm(x, scaling=0.1)

        for rho in [0.1, 1.0, 10.0]:
            config = ADMMConfig(rho=rho)
            solver = ADMM(obj, config)
            stopping_criteria = ADMMStoppingCriteria(
                eps_abs=tol, eps_rel=tol, max_iters=MAX_ITERS
            )
            result = solver.solve(stopping_criteria=stopping_criteria)
            # Should converge regardless of initial rho
            assert result.convergence_status == ConvergenceStatus.CONVERGED

    def test_different_alpha_values(self, reset_torch_state, precision, tol):
        """Test ADMM with different over-relaxation parameters."""
        torch.set_default_dtype(precision)
        A, b, x = generate_least_squares_data(n=128, p=64, precision=precision)
        obj = SumSquares(A @ x - b) + L1Norm(x, scaling=0.1)

        for alpha in [1.0, 1.5, 1.8]:
            config = ADMMConfig(alpha=alpha)
            solver = ADMM(obj, config)
            stopping_criteria = ADMMStoppingCriteria(
                eps_abs=tol, eps_rel=tol, max_iters=MAX_ITERS
            )
            result = solver.solve(stopping_criteria=stopping_criteria)
            assert result.convergence_status == ConvergenceStatus.CONVERGED


# ============================================================================
# Helper Functions
# ============================================================================


def _solve_and_verify(obj, tol):
    """Test that optimization problem is solved correctly."""
    config = ADMMConfig(rho=1.0)
    solver = ADMM(obj, config)

    stopping_criteria = ADMMStoppingCriteria(
        eps_abs=tol, eps_rel=tol, max_iters=MAX_ITERS
    )

    # Test using solve method
    result = solver.solve(stopping_criteria=stopping_criteria)

    # Verify result has expected attributes
    assert result.variable_values is not None
    assert result.num_iters > 0
    assert result.solver_time > 0
    assert hasattr(result, "primal_residual_norm")
    assert hasattr(result, "dual_residual_norm")
    assert result.convergence_status == ConvergenceStatus.CONVERGED
