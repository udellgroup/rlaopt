"""Tests for the proximal gradient solver."""

import pytest
import torch

from rlaopt.atoms import Box, L1Norm, NonNegative, SumSquares
from rlaopt.expression import Variable
from rlaopt.solvers import GradSolverStoppingCriteria, ProxGrad, ProxGradConfig

TOLERANCES = {torch.float32: 1e-4, torch.float64: 1e-7}
MAX_ITERS = 5000


@pytest.fixture(params=[True, False], ids=["accel", "no_accel"])
def acceleration(request):
    """Whether to use acceleration."""
    return request.param


@pytest.fixture(params=[True, False], ids=["linesearch", "no_linesearch"])
def ls(request):
    """Whether to use line search."""
    return request.param


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


def generate_least_squares_data(n=1024, p=256, precision=torch.float32, seed=0):
    """Generate random data for least squares problems."""
    torch.manual_seed(seed)
    A = torch.randn(n, p, dtype=precision) / (n**0.5)
    x_star = torch.randn(p) / (p**0.5)
    b = A @ x_star + 0.01 * torch.randn(n)
    x = Variable(torch.zeros(p, dtype=precision))
    return A, b, x


def generate_lasso_data(n=1024, p=128, s=32, precision=torch.float32, seed=0):
    """Generate random data for LASSO problems with sparse ground truth."""
    torch.manual_seed(seed)

    # Generate sparse ground truth
    J = torch.randperm(p)[:s]
    x_star = torch.zeros(p, dtype=precision)
    x_star[J] = torch.randn(s, dtype=precision) / (s**0.5)

    # Generate measurement matrix and noisy observations
    A = torch.randn(n, p, dtype=precision) / (n**0.5)
    b = A @ x_star + 0.001 * torch.randn(n, dtype=precision)

    x = Variable(torch.zeros(p, dtype=precision))
    return A, b, x, x_star


def generate_matrix_sensing_data(n=64, p=16, precision=torch.float32, seed=0):
    """Generate random data for matrix sensing/completion problems."""
    torch.manual_seed(seed)

    M_star = torch.randn(n, 8, dtype=precision)
    N_star = torch.randn(8, p, dtype=precision)
    X_Star = M_star @ N_star
    A = torch.randn(2 * n, n, dtype=precision)
    B = A @ X_Star + 10**-4 * torch.randn((2 * n, p), dtype=precision)
    X = Variable(torch.zeros_like(X_Star))
    return A, B, X


def compute_lipschitz_stepsize(A, scaling=0.5):
    """Compute stepsize based on Lipschitz constant of gradient."""
    return scaling / (2 * torch.linalg.norm(A, ord=2) ** 2)


# ============================================================================
# Test Class
# ============================================================================


class TestProxGrad:
    """Tests for the proximal gradient solver."""

    def test_least_squares(self, reset_torch_state, precision, acceleration, ls):
        """Test proximal gradient on least squares problem."""
        torch.set_default_dtype(precision)
        A, b, x = generate_least_squares_data(n=1024, p=256, precision=precision)
        obj = SumSquares(A @ x - b)
        eta = compute_lipschitz_stepsize(A)
        _solve_and_verify(obj, eta, acceleration, ls)

    def test_box(self, reset_torch_state, precision, acceleration, ls):
        """Test proximal gradient on box-constrained problem."""
        torch.set_default_dtype(precision)
        A, b, x = generate_least_squares_data(n=1024, p=256, precision=precision)
        lower = -2.0
        upper = 1.0
        obj = SumSquares(A @ x - b) + Box(x, lower=lower, upper=upper)
        eta = compute_lipschitz_stepsize(A)
        _solve_and_verify(obj, eta, acceleration, ls)

    def test_nonnegative(self, reset_torch_state, precision, acceleration, ls):
        """Test proximal gradient on nonnegative-constrained problem."""
        torch.set_default_dtype(precision)
        A, b, x = generate_least_squares_data(n=1024, p=256, precision=precision)
        obj = SumSquares(A @ x - b) + NonNegative(x)
        eta = compute_lipschitz_stepsize(A)
        _solve_and_verify(obj, eta, acceleration, ls)

    def test_lasso(self, reset_torch_state, precision, acceleration, ls):
        """Test proximal gradient on LASSO problem."""
        torch.set_default_dtype(precision)
        A, b, x, _ = generate_lasso_data(n=1024, p=128, s=32, precision=precision)
        mu = 0.1 * torch.linalg.norm(A.T @ b, ord=torch.inf)
        obj = SumSquares(A @ x - b) + L1Norm(x, scaling=mu)
        eta = compute_lipschitz_stepsize(A)
        _solve_and_verify(obj, eta, acceleration, ls)

    # def test_nucnorm(self, reset_torch_state, precision, tol, acceleration, ls):
    #     """Test proximal gradient on nuclear norm regularized problem."""
    #     torch.set_default_dtype(precision)
    #     A, B, X = generate_matrix_sensing_data(n=64, p=16, precision=precision)
    #     lambd = 1000.0
    #     obj = SumSquares(A @ X - B) + NucNorm(X, scaling=lambd)
    #     eta = compute_lipschitz_stepsize(A)
    #     _solve_and_verify(obj, eta, tol, acceleration, ls)

    def test_nonsmooth_subset_of_smooth(
        self, reset_torch_state, precision, acceleration, ls
    ):
        """Test when non-smooth variables are subset of smooth variables."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 256, 64
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        y = Variable(torch.zeros(32, dtype=precision), name="y")
        lambd = 0.1
        obj = SumSquares(A @ x - b) + SumSquares(y) + L1Norm(x, scaling=lambd)
        # Hessian has A.T @ A for x block and 2*I for y block
        # Use more conservative stepsize accounting for both blocks
        eta = compute_lipschitz_stepsize(A, scaling=0.25)
        _solve_and_verify(obj, eta, acceleration, ls)

    def test_nonsmooth_disjoint_from_smooth(
        self, reset_torch_state, precision, acceleration, ls
    ):
        """Test non-smooth variables disjoint from smooth variables."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 256, 64
        A = torch.randn(n, p, dtype=precision) / (n**0.5)
        b = torch.randn(n, dtype=precision) / (n**0.5)
        x = Variable(torch.zeros(p, dtype=precision), name="x")
        z = Variable(torch.zeros(32, dtype=precision), name="z")
        obj = SumSquares(A @ x - b) + L1Norm(z, scaling=0.1)
        eta = compute_lipschitz_stepsize(A)
        _solve_and_verify(obj, eta, acceleration, ls)

    def test_complex_mixed_variables(
        self, reset_torch_state, precision, acceleration, ls
    ):
        """Test complex case with multiple smooth and non-smooth terms."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        n, p = 128, 32
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
        # Hessian has A.T @ A blocks and I blocks from SumSquares(x+y)
        # Use more conservative stepsize
        eta = compute_lipschitz_stepsize(A, scaling=0.5)
        _solve_and_verify(obj, eta, acceleration, ls)

    def test_pure_nonsmooth(self, reset_torch_state, precision, acceleration, ls):
        """Test pure non-smooth objective (no smooth part)."""
        torch.set_default_dtype(precision)
        torch.manual_seed(0)
        p = 64
        x = Variable(torch.randn(p, dtype=precision), name="x")
        obj = L1Norm(x)
        eta = 1.0  # Stepsize doesn't affect pure prox
        _solve_and_verify(obj, eta, acceleration, ls)


class TestProxGradErrors:
    """Tests for error conditions in ProxGrad initialization."""

    def test_invalid_objective_type(self):
        """Test that ProxGrad raises ValueError for non-Expression objective."""
        config = ProxGradConfig()
        with pytest.raises(ValueError, match="Expression objective"):
            ProxGrad("not an expression", config)

    def test_invalid_config_type(self):
        """Test that ProxGrad raises ValueError for non-ProxGradConfig."""
        x = Variable(torch.ones(5), name="x")
        obj = SumSquares(x)
        with pytest.raises(ValueError, match="ProxGradConfig configuration"):
            ProxGrad(obj, "not a config")


# ============================================================================
# Helper Functions
# ============================================================================


def _solve_and_verify(obj, eta, use_acceleration, use_linesearch):
    """Test that optimization problem is solved correctly."""
    opt = _build_opt(obj, eta, use_acceleration, use_linesearch)
    stopping_criteria = GradSolverStoppingCriteria(max_iters=MAX_ITERS)

    # Test using solve method
    results = opt.solve(stopping_criteria=stopping_criteria)
    assert results.convergence_status.value == "converged"


def _init_opt(obj, opt):
    """Initialize optimizer state."""
    params = obj.variable_values
    return params, opt.init_state(params)


def _build_opt(obj, eta, use_acceleration, use_linesearch):
    """Build proximal gradient optimizer with specified configuration."""
    config = ProxGradConfig(
        eta=eta,
        use_acceleration=use_acceleration,
        use_linesearch=use_linesearch,
    )
    return ProxGrad(obj, config)
