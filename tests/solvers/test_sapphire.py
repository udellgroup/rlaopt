"""Tests for the SAPPHIRE solver."""

import pytest
import torch

from rlaopt.atoms import (
    Box,
    L1Norm,
    LinearRegression,
    LogisticRegression,
    NonNegative,
    SumSquares,
)
from rlaopt.data import DataLoader, Dataset
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import IdentityConfig, NystromConfig
from rlaopt.linalg.preconditioners.nystrom import Nystrom
from rlaopt.solvers import GradSolverStoppingCriteria, Sapphire, SapphireConfig
from rlaopt.solvers.gradient_solvers.gradient_solver_core import prox_update_P
from rlaopt.solvers.gradient_solvers.gradient_solver_states import SGDState
from rlaopt.splitting.sapphire_split import SapphireSplit

MAX_ITERS = 3000


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=["saga", "svrg"], ids=["saga_base", "svrg_base"])
def base(request):
    """Base method to use in SAPPHIRE."""
    return request.param


@pytest.fixture(
    params=["identity", "nystrom"], ids=["identity_precond", "nystrom_precond"]
)
def config(request, base):
    """Solver config for SAPPHIRE."""
    if request.param == "identity":
        return SapphireConfig(
            eta=1.0,
            base_method=base,
            precond_config=IdentityConfig(),
            auto_update_stepsize=False,
        )
    else:
        # NystromConfig is the default preconditioner
        return SapphireConfig(
            base_method=base, auto_update_stepsize=True, precond_update_freq=2
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


def generate_log_reg_data(batch_size, n=1024, p=256, precision=torch.float32, seed=0):
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


def generate_linear_reg_data(
    batch_size,
    n=1024,
    p=64,
    precision=torch.float32,
    seed=0,
):
    """Generate noiseless least-squares data (interpolation regime)."""
    torch.manual_seed(seed)

    X = torch.randn(n, p, dtype=precision)
    beta_star = torch.randn(p, dtype=precision)
    y = X @ beta_star  # no noise — interpolation is possible

    beta = Variable(torch.zeros(p, dtype=precision), name="beta")
    dataset = Dataset(X, y, dtype=precision)
    loader = DataLoader(dataset, batch_size=batch_size)

    return beta, loader


def generate_lasso_data(
    batch_size, n=1024, p=128, s=32, precision=torch.float32, seed=0
):
    """Generate random data for LASSO problems with sparse ground truth."""
    torch.manual_seed(seed)

    # Generate sparse ground truth
    J = torch.randperm(p)[:s]
    beta_star = torch.zeros(p, dtype=precision)
    beta_star[J] = torch.randn(s, dtype=precision)

    # Generate measurement matrix and noisy observations
    X = torch.randn(n, p, dtype=precision) / (n**0.5)
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
        obj = LogisticRegression(beta, loader) + 0.001 * SumSquares(beta)
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


@pytest.fixture(
    params=["identity", "nystrom"],
    ids=["identity_precond", "nystrom_precond"],
)
def sgd_config(request):
    """Solver config for SGD-base SAPPHIRE tests."""
    if request.param == "identity":
        return SapphireConfig(
            base_method="sgd",
            eta=0.1,
            precond_config=IdentityConfig(),
            auto_update_stepsize=False,
        )
    else:
        return SapphireConfig(
            base_method="sgd",
            auto_update_stepsize=True,
            precond_update_freq=2,
        )


# ============================================================================
# Solver Mechanism Tests
# ============================================================================


class TestSapphireMechanisms:
    """Tests for internal solver mechanisms (step-size, state bookkeeping)."""

    def test_auto_update_stepsize(self, batch_size, precision):
        """Step-size should change after the first step with auto_update_stepsize."""
        torch.set_default_dtype(precision)
        beta, loader = generate_log_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=0
        )
        obj = LogisticRegression(beta, loader) + 0.001 * SumSquares(beta)
        config = SapphireConfig(base_method="saga", auto_update_stepsize=True)
        opt = Sapphire(obj, config)

        init_state = opt.init_state(opt._op_split.variable_values)
        init_eta = init_state.eta

        _, new_state = opt.step(opt._op_split.variable_values, init_state)

        assert new_state.eta != init_eta, (
            f"eta should have been updated by auto step-size but stayed at {init_eta}"
        )

    def test_saga_table_updates(self, batch_size, precision):
        """SAGA gradient table and average should be non-zero after the first step."""
        torch.set_default_dtype(precision)
        beta, loader = generate_log_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=1
        )
        obj = LogisticRegression(beta, loader) + 0.001 * SumSquares(beta)
        config = SapphireConfig(
            base_method="saga",
            precond_config=IdentityConfig(),
            auto_update_stepsize=False,
        )
        opt = Sapphire(obj, config)

        var_values = opt._op_split.variable_values
        state = opt.init_state(var_values)

        assert state.table is not None
        assert torch.all(state.table == 0), "SAGA table should be zeros at init"
        assert torch.all(state.grad_avg["beta"] == 0), "grad_avg should be zero at init"

        var_values, new_state = opt.step(var_values, state)

        assert not torch.all(new_state.table == 0), (
            "SAGA table should be updated after a step"
        )
        assert not torch.all(new_state.grad_avg["beta"] == 0), (
            "grad_avg should be non-zero after a step"
        )

    def test_svrg_snapshot_updates(self, batch_size, precision):
        """SVRG snapshot should be None initially and populated after enough steps."""
        torch.set_default_dtype(precision)
        beta, loader = generate_log_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=2
        )
        obj = LogisticRegression(beta, loader) + 0.001 * SumSquares(beta)
        config = SapphireConfig(
            base_method="svrg",
            precond_config=IdentityConfig(),
            auto_update_stepsize=False,
            snapshot_update_freq=1,
        )
        opt = Sapphire(obj, config)

        n = opt._op_split.num_samples
        grad_batch_size = loader.batch_size
        # Number of steps before the first snapshot update fires
        snapshot_threshold = (n // grad_batch_size) * config.snapshot_update_freq

        var_values = opt._op_split.variable_values
        state = opt.init_state(var_values)

        assert state.snapshot is None
        assert state.snapshot_grad is None

        for _ in range(snapshot_threshold):
            var_values, state = opt.step(var_values, state)

        assert state.snapshot is not None, (
            "Snapshot should be set after threshold steps"
        )
        assert state.snapshot_grad is not None, (
            "Snapshot grad should be set after threshold steps"
        )

        # Take another epoch's worth of steps and verify snapshot actually changes
        snapshot_before = state.snapshot["beta"].clone()
        for _ in range(snapshot_threshold):
            var_values, state = opt.step(var_values, state)

        assert not torch.allclose(state.snapshot["beta"], snapshot_before), (
            "Snapshot should update on subsequent snapshot intervals"
        )


# ============================================================================
# SGD Tests
# ============================================================================


class TestSapphireSGD:
    """Tests specific to the SGD base method."""

    def test_sgd_convergence_interpolation(self, batch_size, precision, sgd_config):
        """SGD converges on noiseless least-squares where zero loss is achievable."""
        torch.set_default_dtype(precision)
        beta, loader = generate_linear_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=10
        )
        obj = LinearRegression(beta, loader, fit_intercept=False)
        _solve_and_verify(obj, sgd_config, seed=10)

    def test_sgd_loss_reduction(self, batch_size):
        """SGD reduces the loss over one epoch on a logistic regression problem."""
        precision = torch.float32
        torch.set_default_dtype(precision)
        beta, loader = generate_log_reg_data(
            batch_size, n=1024, p=64, precision=precision, seed=11
        )
        obj = LogisticRegression(beta, loader)
        config = SapphireConfig(
            base_method="sgd",
            eta=0.1,
            precond_config=IdentityConfig(),
            auto_update_stepsize=False,
        )
        opt = Sapphire(obj, config)

        var_values = opt._op_split.variable_values
        state = opt.init_state(var_values)

        loss_before = opt._op_split.evaluate(var_values)

        steps_per_epoch = opt._op_split.num_samples // loader.batch_size
        for _ in range(steps_per_epoch):
            var_values, state = opt.step(var_values, state)

        loss_after = opt._op_split.evaluate(var_values)

        assert loss_after < loss_before, (
            f"SGD should reduce loss after one epoch, but loss went from "
            f"{loss_before:.6f} to {loss_after:.6f}"
        )


# ============================================================================
# Differentiability Tests
# ============================================================================


class TestSapphireDifferentiability:
    """Tests that Sapphire supports differentiating through the solver."""

    def test_diff_through_lasso(self):
        """Gradient of test loss w.r.t. L1 penalty mu should be finite and non-zero."""
        torch.manual_seed(0)
        precision = torch.float32
        torch.set_default_dtype(precision)

        n, p = 512, 32
        n_train = int(0.8 * n)

        # Generate sparse logistic regression data
        X = torch.randn(n, p, dtype=precision)
        beta_star = torch.zeros(p, dtype=precision)
        J = torch.randperm(p)[:8]
        beta_star[J] = torch.randn(8, dtype=precision)
        y = (torch.sigmoid(X @ beta_star) > 0.5).float()

        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        dataset_train = Dataset(X_train, y_train, dtype=precision)
        loader_train = DataLoader(dataset_train, batch_size=64)

        def test_loss(mu: torch.Tensor) -> torch.Tensor:
            beta = Variable(torch.zeros(p, dtype=precision), name="beta")
            obj = LogisticRegression(beta, loader_train) + L1Norm(beta, scaling=mu)
            config = SapphireConfig(
                base_method="saga",
                eta=0.1,
                precond_config=IdentityConfig(),
                auto_update_stepsize=False,
            )
            opt = Sapphire(obj, config, detach=False)
            result = opt.solve(
                stopping_criteria=GradSolverStoppingCriteria(max_iters=500)
            )
            beta_sol = result.variable_values["beta"]

            # Test-set cross-entropy loss
            logits = X_test @ beta_sol
            log_p = torch.nn.functional.logsigmoid(logits)
            return (
                -1
                / X_test.shape[0]
                * torch.sum(y_test * log_p + (1 - y_test) * (-logits + log_p))
            )

        mu = torch.tensor(0.01, dtype=precision)
        grad, _ = torch.func.grad_and_value(test_loss)(mu)

        assert torch.isfinite(grad), f"Gradient w.r.t. mu should be finite, got {grad}"
        assert grad != 0.0, "Gradient w.r.t. mu should be non-zero"

        eps = torch.tensor(1e-3, dtype=mu.dtype)
        numerical_grad = (test_loss(mu + eps) - test_loss(mu - eps)) / (2 * eps)
        assert torch.allclose(grad, numerical_grad, rtol=0.05), (
            f"Autodiff grad {grad:.4f} does not match "
            f"numerical grad {numerical_grad:.4f}"
        )


# ============================================================================
# prox_update_P Tests
# ============================================================================


class TestProxUpdateP:
    """Tests that prox_update_P correctly solves the preconditioned proximal subproblem.

    prox_update_P runs APG on:
        minimize_z  (1/2)(z - x_k)^T P (z - x_k) + eta * grads^T(z - x_k) + eta * r(z)

    At convergence the gradient mapping norm should be approximately zero and,
    for constrained problems, the output should satisfy the constraints.
    """

    @pytest.fixture
    def nystrom_state(self):
        """Build a realistic Nystrom P from a logistic regression Hessian."""
        torch.manual_seed(0)
        precision = torch.float32
        n, p = 64, 16

        X = torch.randn(n, p, dtype=precision)
        beta_star = torch.randn(p, dtype=precision)
        y = (torch.sigmoid(X @ beta_star) > 0.5).float()

        beta = Variable(torch.zeros(p, dtype=precision), name="beta")
        dataset = Dataset(X, y, dtype=precision)
        loader = DataLoader(dataset, batch_size=n)
        model = LogisticRegression(beta, loader, fit_intercept=False)
        split = SapphireSplit(model)

        # Build Nystrom P from full-batch Hessian
        Hop = split.get_subsamp_hessian_linop(model.variable_values, X, y, X.device)
        P = Nystrom(
            NystromConfig(
                rank_init=8,
                rank_max=16,
                error_tolerance=1e-1,
                base_damping=1e-2,
                damping_mode="adaptive",
            )
        )
        P._update(Hop, precision)

        # Random current iterate and gradient direction
        var_vals = TensorDict({"beta": torch.randn(p, dtype=precision)})
        grads = TensorDict({"beta": torch.randn(p, dtype=precision)})
        state = SGDState(iter_=0, eta=0.1, P=P)

        return var_vals, grads, state, p, precision

    @staticmethod
    def _grad_mapping_norm(z, var_vals, grads, state, prox_fn):
        """Gradient mapping norm for the preconditioned proximal subproblem at z.

        G(z) = ||P||*||z - prox_{(eta/||P||)*r}(z - (1/||P||)*(P(z-x_k)+eta*grads))||
        """
        alpha = 1.0 / state.P.norm
        Pd = z.from_flat_tensor(state.P @ (z - var_vals).to_flat_tensor())
        z_int = z - alpha * (state.eta * grads + Pd)
        z_prox = prox_fn(z_int.clone(), state.eta * alpha)
        return (z.to_flat_tensor() - z_prox.to_flat_tensor()).norm().item() / alpha

    def test_box_constraint(self, nystrom_state):
        """After 200 APG iters, gradient mapping ≈ 0 and box constraints satisfied."""
        var_vals, grads, state, p, precision = nystrom_state
        beta = Variable(torch.zeros(p, dtype=precision), name="beta")
        r = Box(beta, lower=-1.0, upper=1.0)

        z, _ = prox_update_P(
            var_vals, grads, state, subproblem_iters=200, prox_fn=r.prox
        )

        # Constraint satisfaction (momentum in last APG step can push z slightly
        # outside the box by O(gradient mapping norm), so allow 1e-4 slack)
        assert torch.all(z["beta"] >= -1.0 - 1e-4), "Output violates lower box bound"
        assert torch.all(z["beta"] <= 1.0 + 1e-4), "Output violates upper box bound"

        gm = self._grad_mapping_norm(z, var_vals, grads, state, r.prox)
        assert gm < 1e-4, f"Gradient mapping norm {gm:.2e} exceeds tolerance"

    def test_l1_regularizer(self, nystrom_state):
        """After 200 APG iters, gradient mapping ≈ 0 for L1-regularized subproblem."""
        var_vals, grads, state, p, precision = nystrom_state
        beta = Variable(torch.zeros(p, dtype=precision), name="beta")
        r = L1Norm(beta, scaling=0.1)

        z, _ = prox_update_P(
            var_vals, grads, state, subproblem_iters=200, prox_fn=r.prox
        )

        gm = self._grad_mapping_norm(z, var_vals, grads, state, r.prox)
        assert gm < 1e-4, f"Gradient mapping norm {gm:.2e} exceeds tolerance"


# ============================================================================
# Helper Functions
# ============================================================================


def _solve_and_verify(obj, config, seed):
    """Test that optimization problem is solved correctly."""
    torch.manual_seed(seed)
    opt = _build_opt(obj, config)
    stopping_criteria = GradSolverStoppingCriteria(max_iters=MAX_ITERS)

    # Test using solve method
    results = opt.solve(stopping_criteria=stopping_criteria)
    assert results.convergence_status.value == "converged"


def _build_opt(obj, config):
    """Build SAPPHIRE optimizer with specified configuration."""
    return Sapphire(obj, config)
