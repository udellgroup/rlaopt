"""Test OperatorSplit class."""

import pytest
import torch

from rlaopt.atoms import Affine, L1Norm, SumSquares
from rlaopt.expression import ProductExpression, Variable
from rlaopt.operator_split import OperatorSplit


@pytest.fixture
def least_squares_data():
    """Generate random data for least squares problems."""
    n, p = 32, 16
    torch.manual_seed(0)
    A = torch.randn(n, p, dtype=torch.float32) / (n**0.5)
    b = torch.randn(n, dtype=torch.float32) / (n**0.5)
    x = Variable(torch.ones(p, dtype=torch.float32), name="x")
    return A, b, x


@pytest.fixture
def smooth_only_problem(least_squares_data):
    """Setup smooth-only optimization problem."""
    A, b, x = least_squares_data
    f = 0.5 * SumSquares(A @ x - b)
    obj = OperatorSplit(f)
    return obj, A, b, x


@pytest.fixture
def lasso_problem(least_squares_data):
    """Setup Lasso (smooth + non-smooth) optimization problem."""
    A, b, x = least_squares_data
    f = 0.5 * SumSquares(A @ x - b)
    r = L1Norm(x)
    obj = OperatorSplit(f, r)
    return obj, A, b, x, r


class TestOperatorSplitInitialization:
    """Tests for OperatorSplit initialization."""

    def test_smooth_only_initialization(self, smooth_only_problem):
        """Test initialization with smooth function only."""
        obj, _, _, _ = smooth_only_problem
        assert isinstance(obj.f, ProductExpression)
        assert obj.r is None

    def test_lasso_initialization(self, lasso_problem):
        """Test initialization with smooth and non-smooth functions."""
        obj, _, _, _, r = lasso_problem
        assert isinstance(obj.f, ProductExpression)
        assert isinstance(obj.r, L1Norm)

    def test_invalid_smooth_function_raises_error(self, least_squares_data):
        """Test that non-smooth expression for smooth function raises ValueError."""
        _, _, x = least_squares_data
        r = L1Norm(x)
        with pytest.raises(ValueError):
            OperatorSplit(r)

    def test_non_proxable_nonsmooth_raises_error(self, least_squares_data):
        """Test that non-proxable non-smooth term raises ValueError."""
        A, b, x = least_squares_data
        f = 0.5 * SumSquares(A @ x - b)
        r = L1Norm(x)
        non_proxable = Affine(r, torch.tensor(1.0), torch.tensor(0.0))
        with pytest.raises(ValueError):
            OperatorSplit(f, non_proxable)


class TestOperatorSplitOracles:
    """Tests for OperatorSplit function and gradient oracles."""

    def test_function_evaluation(self, smooth_only_problem):
        """Test function value computation."""
        obj, A, b, x = smooth_only_problem
        p = A.shape[1]
        new_params = obj.f.params_from_tensors((torch.zeros(p),))
        expected = 0.5 * torch.linalg.norm(b) ** 2
        assert torch.allclose(obj.f_func(new_params), expected)

    def test_gradient_computation(self, smooth_only_problem):
        """Test gradient computation."""
        obj, A, b, x = smooth_only_problem
        p = A.shape[1]
        new_params = obj.f.params_from_tensors((torch.zeros(p),))
        name = obj.f.params_names()[0]
        grads = obj.grad_f(new_params)
        assert torch.allclose(grads[name], -A.T @ b)

    def test_hvp_computation(self, smooth_only_problem):
        """Test Hessian-vector product computation."""
        obj, A, b, x = smooth_only_problem
        p = A.shape[1]
        new_params = obj.f.params_from_tensors((torch.zeros(p),))
        v = torch.ones(p)
        Hv = obj.hvp_f(new_params, v)
        assert torch.allclose(Hv, A.T @ (A @ v))


class TestOperatorSplitProx:
    """Tests for OperatorSplit proximal operator."""

    def test_prox_smooth_only_unchanged(self, smooth_only_problem):
        """Test prox returns input unchanged for smooth-only problem."""
        obj, _, _, x = smooth_only_problem
        name = obj.f.params_names()[0]
        result = obj.prox(obj.f.params, 1.0)
        assert torch.allclose(result[name], x.value)

    def test_prox_lasso_matches_l1norm(self, lasso_problem):
        """Test prox corresponds to L1Norm prox for Lasso problem."""
        obj, _, _, x, r = lasso_problem
        obj_prox_result = obj.prox(obj.r.params, 1.0)["x"]
        expected = r.prox(x.value, 1.0)
        assert torch.allclose(obj_prox_result, expected)
