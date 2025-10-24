"""Tests for the proximal gradient solver."""

import pytest
import torch

from rlaopt.atoms import Box, L1Norm, NonNegative, SumSquares
from rlaopt.expression.expression import Variable
from rlaopt.solvers.prox_grad import ProxGrad, ProxGradConfig, ProxGradStoppingCriteria
from rlaopt.utils import tensor_dict_ops as dict_ops

TOLERANCES = {torch.float32: 1e-4, torch.float64: 1e-10}
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
    b = torch.randn(n, dtype=precision) / (n**0.5)
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
    return scaling / (torch.linalg.norm(A, ord=2) ** 2)


# ============================================================================
# Test Class
# ============================================================================


class TestProxGrad:
    """Tests for the proximal gradient solver."""

    def test_least_squares(self, reset_torch_state, precision, tol, acceleration, ls):
        """Test proximal gradient on least squares problem."""

        def setup_problem():
            A, b, x = generate_least_squares_data(n=1024, p=256, precision=precision)
            obj = SumSquares(A @ x - b)
            return A, obj

        _test_optimization_problem(precision, tol, acceleration, ls, setup_problem)

    def test_box(self, reset_torch_state, precision, tol, acceleration, ls):
        """Test proximal gradient on box-constrained problem."""

        def setup_problem():
            A, b, x = generate_least_squares_data(n=1024, p=256, precision=precision)
            lower = -torch.tensor(2.0)
            upper = torch.tensor(1.0)
            obj = SumSquares(A @ x - b) + Box(x, l=lower, u=upper)
            return A, obj

        _test_optimization_problem(precision, tol, acceleration, ls, setup_problem)

    def test_nonnegative(self, reset_torch_state, precision, tol, acceleration, ls):
        """Test proximal gradient on nonnegative-constrained problem."""

        def setup_problem():
            A, b, x = generate_least_squares_data(n=1024, p=256, precision=precision)
            obj = SumSquares(A @ x - b) + NonNegative(x)
            return A, obj

        _test_optimization_problem(precision, tol, acceleration, ls, setup_problem)

    def test_lasso(self, reset_torch_state, precision, tol, acceleration, ls):
        """Test proximal gradient on LASSO problem."""

        def setup_problem():
            A, b, x, _ = generate_lasso_data(n=1024, p=128, s=32, precision=precision)
            mu = 0.1 * torch.linalg.norm(A.T @ b, ord=torch.inf)
            obj = SumSquares(A @ x - b) + L1Norm(x, scaling=mu)
            return A, obj

        _test_optimization_problem(precision, tol, acceleration, ls, setup_problem)

    # def test_nucnorm(self, reset_torch_state, precision, tol, acceleration, ls):
    #     """Test proximal gradient on nuclear norm regularized problem."""
    #     def setup_problem():
    #         A, B, X = generate_matrix_sensing_data(n=64, p=16, precision=precision)
    #         lambd = 1000.0
    #         obj = SumSquares(A @ X - B) + NucNorm(X, scaling=lambd)
    #         return A, obj

    #     _test_optimization_problem(precision, tol, acceleration, ls, setup_problem)


# ============================================================================
# Helper Functions
# ============================================================================


def _test_optimization_problem(precision, tol, acceleration, ls, setup_fn):
    """Common test structure for all optimization problems."""
    torch.set_default_dtype(precision)
    A, obj = setup_fn()
    eta = compute_lipschitz_stepsize(A)
    _solve_and_verify(obj, eta, tol, acceleration, ls)


def _solve_and_verify(obj, eta, tol, use_acceleration, use_linesearch):
    """Test that optimization problem is solved correctly."""
    opt = _build_opt(obj, eta, use_acceleration, use_linesearch)
    stopping_criteria = ProxGradStoppingCriteria(tol=tol, max_iters=MAX_ITERS)
    params, state = _init_opt(obj, opt)

    # Test solving by step
    params, err = _loop(params, state, opt, stopping_criteria)
    assert err <= tol, f"Step-by-step solving failed: error {err} > tolerance {tol}"

    # Test using solve method
    params, err = opt.solve(stopping_criteria=stopping_criteria)
    assert err.item() <= tol * (dict_ops.dim(params) ** 0.5), (
        f"Solve method failed: error {err.item()} > tolerance {tol}"
    )


def _loop(params, state, opt, stopping_criteria):
    """Run optimization loop until convergence or max iterations."""
    for _ in range(stopping_criteria.max_iters):
        params, state = opt.step(params, state)
        if state.err.item() <= stopping_criteria.tol:
            break
    return params, state.err.item()


def _init_opt(obj, opt):
    """Initialize optimizer state."""
    params = obj.params
    return params, opt.init_state(params)


def _build_opt(obj, eta, use_acceleration, use_linesearch):
    """Build proximal gradient optimizer with specified configuration."""
    config = ProxGradConfig(
        eta=eta,
        use_acceleration=use_acceleration,
        use_linesearch=use_linesearch,
    )
    return ProxGrad(obj, config)
