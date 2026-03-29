"""Tests for the SAPPHIRE solver."""

import pytest
import torch

from rlaopt.atoms import Box, LogisticRegression, L1Norm, NonNegative, SumSquares
from rlaopt.data import Dataset, DataLoader
from rlaopt.expression import Variable
from rlaopt.linalg import IdentityConfig
from rlaopt.solvers import GradSolverStoppingCriteria, SapphireConfig, Sapphire


MAX_ITERS = 3000


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=["saga", "svrg"], ids=["saga_base", "svrg_base"])
def base(request):
    """Base method to use in SAPPHIRE."""
    return request.param


@pytest.fixture(params=["identity", "nystrom"], ids=["identity_precond", "nystrom_precond"])
def config(request, base):
    """Solver config for SAPPHIRE."""
    if request.param == "identity":
        return SapphireConfig(
            eta=1.0,
            base_method=base,
            precond_config=IdentityConfig(),
            auto_update_stepsize=False
        )
    else:
        # NystromConfig is the default preconditioner
        return SapphireConfig(
            base_method=base,
            auto_update_stepsize=True,
            precond_update_freq=2
        )


@pytest.fixture
def batch_size():
    """Batch size to use in SAPPHIRE."""
    return 256


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    """Torch dtype for the test."""
    return request.param



@pytest.fixture(autouse=True)
def reset_torch_state():
    """Fixture to reset torch default dtype before and after each test."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(original_dtype)
    yield
    torch.set_default_dtype(original_dtype)


# ============================================================================
# Data Generation Helpers
# ============================================================================


def generate_log_reg_data(
    batch_size,
    n=1024,
    p=256,
    precision=torch.float32,
    seed=0
):
    """Generate random data for least squares problems."""
    torch.manual_seed(seed)
    
    X = torch.randn(n, p, dtype=precision)
    beta_star = torch.randn(p, dtype=precision)
    logits = X @ beta_star + 0.01 * torch.randn(n, dtype=precision)
    probs = torch.sigmoid(logits)
    y = (probs > 0.5).float()
    
    beta = Variable(torch.zeros(p, dtype=precision), name="beta")
    dataset = Dataset(X, y, dtype=precision)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    return beta, loader


def generate_lasso_data(
    batch_size,
    n=1024,
    p=128,
    s=32,
    precision=torch.float32,
    seed=0
):
    """Generate random data for LASSO problems with sparse ground truth."""
    torch.manual_seed(seed)

    # Generate sparse ground truth
    J = torch.randperm(p)[:s]
    beta_star = torch.zeros(p, dtype=precision)
    beta_star[J] = torch.randn(s, dtype=precision)

    # Generate measurement matrix and noisy observations
    X = torch.randn(n, p, dtype=precision) / (n ** 0.5)
    logits = X @ beta_star + 0.01 * torch.randn(n, dtype=precision)
    probs = torch.sigmoid(logits)
    y = (probs > 0.5).float()

    beta = Variable(torch.zeros(p, dtype=precision), name="beta")
    dataset = Dataset(X, y, dtype=precision)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    return beta, loader


# ============================================================================
# Test Class
# ============================================================================


class TestSapphire:
    """Tests for the SAPPHIRE solver."""

    def test_ridge(self, batch_size, precision, config):
        """Test SAPPHIRE on logistic regression problem with ridge regularization."""
        torch.set_default_dtype(precision)
        beta, loader = generate_log_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=42
        )
        obj = LogisticRegression(beta, loader) +  0.001 * SumSquares(beta)
        _solve_and_verify(obj, config, seed=42)

    def test_box(self, batch_size, precision, config):
        """Test SAPPHIRE on box-constrained problem."""
        torch.set_default_dtype(precision)
        beta, loader = generate_log_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=43
        )
        obj = LogisticRegression(beta, loader) + Box(beta, lower=-2.0, upper=1.0)
        _solve_and_verify(obj, config, seed=43)

    def test_nonnegative(self, batch_size, precision, config):
        """Test SAPPHIRE on nonnegative-constrained problem."""
        torch.set_default_dtype(precision)
        beta, loader = generate_log_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=44
        )
        obj = LogisticRegression(beta, loader) + NonNegative(beta)
        _solve_and_verify(obj, config, seed=44)

    def test_lasso(self, batch_size, precision, config):
        """Test SAPPHIRE on LASSO problem."""
        torch.set_default_dtype(precision)
        beta, loader = generate_lasso_data(
            batch_size, n=1024, p=128, s=32, precision=precision, seed=45
        )
        mu = 0.005
        obj = LogisticRegression(beta, loader) + L1Norm(beta, scaling=mu)
        _solve_and_verify(obj, config, seed=45)


# ============================================================================
# Helper Functions
# ============================================================================


def _solve_and_verify(obj, config, seed):
    """Test that optimization problem is solved correctly."""
    torch.manual_seed(seed)
    opt = _build_opt(obj, config)
    stopping_criteria = GradSolverStoppingCriteria(
        max_iters=MAX_ITERS
    )

    # Test using solve method
    results = opt.solve(stopping_criteria=stopping_criteria)
    assert results.convergence_status.value == "converged"


def _build_opt(obj, config):
    """Build SAPPHIRE optimizer with specified configuration."""
    return Sapphire(obj, config)