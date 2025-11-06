"""Test OperatorSplit class."""

import pytest
import torch

from rlaopt.atoms import Affine, L1Norm, NonNegative, SumSquares
from rlaopt.expression import ProductExpression, Variable
from rlaopt.ext_tensordict import TensorDict
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


@pytest.fixture
def mult_params_problem():
    """Problem with multiple parameters."""
    x = Variable(torch.ones(10), name="x")
    Y = Variable(torch.zeros(10, 5), name="Y")
    f = 0.5 * SumSquares(x) + SumSquares(Y)
    obj = OperatorSplit(f)
    return obj, x, Y


class TestOperatorSplitInitialization:
    """Tests for OperatorSplit initialization."""

    def test_smooth_only_initialization(self, smooth_only_problem):
        """Test initialization with smooth function only."""
        obj, _, _, _ = smooth_only_problem
        assert obj._f is not None
        assert obj._r is None
        assert isinstance(obj.f, ProductExpression)
        assert obj.r is None

    def test_lasso_initialization(self, lasso_problem):
        """Test initialization with smooth and non-smooth functions."""
        obj, _, _, _, _ = lasso_problem
        assert obj._f is not None
        assert obj._r is not None
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

    def test_evaluate_smooth_only(self, smooth_only_problem):
        """Test evaluate method with smooth-only problem."""
        obj, A, b, _ = smooth_only_problem
        p = A.shape[1]
        new_params = TensorDict({"x": torch.zeros(p)})
        expected = 0.5 * torch.linalg.norm(b) ** 2
        assert torch.allclose(obj.evaluate(new_params), expected)

    def test_evaluate_with_nonsmooth(self, lasso_problem):
        """Test evaluate method with smooth + nonsmooth problem."""
        obj, A, b, x, r = lasso_problem
        p = A.shape[1]
        test_values = torch.tensor([1.0, -2.0, 3.0] + [0.0] * (p - 3))
        new_params = TensorDict({"x": test_values})

        # Compute expected: f + r
        f_val = 0.5 * torch.sum((A @ test_values - b) ** 2)
        r_val = torch.sum(torch.abs(test_values))
        expected = f_val + r_val

        assert torch.allclose(obj.evaluate(new_params), expected)

    def test_evaluate_multiple_params(self, mult_params_problem):
        """Test evaluate with multiple parameters."""
        obj, x, Y = mult_params_problem
        p1 = x.value.shape[0]
        p2, p3 = Y.value.shape
        new_params = TensorDict({"x": -torch.ones(p1), "Y": torch.ones(p2, p3)})

        # Expected: 0.5 * sum(x^2) + sum(Y^2) = 0.5 * p1 + p2 * p3
        expected = torch.tensor(float(0.5 * p1 + p2 * p3))
        assert torch.allclose(obj.evaluate(new_params), expected)

    def test_function_evaluation(self, smooth_only_problem):
        """Test function value computation."""
        obj, A, b, _ = smooth_only_problem
        p = A.shape[1]
        new_params = TensorDict({"x": torch.zeros(p)})
        expected = 0.5 * torch.linalg.norm(b) ** 2
        assert torch.allclose(obj.f_func(new_params), expected)

    def test_gradient_computation(self, smooth_only_problem):
        """Test gradient computation."""
        obj, A, b, _ = smooth_only_problem
        p = A.shape[1]
        new_params = TensorDict({"x": torch.zeros(p)})
        grads = obj.grad_f(new_params)
        assert torch.allclose(grads["x"], -A.T @ b)

    def test_hvp_computation(self, smooth_only_problem):
        """Test Hessian-vector product computation."""
        obj, A, _, _ = smooth_only_problem
        p = A.shape[1]
        new_params = TensorDict({"x": torch.zeros(p)})
        v = torch.ones(p)
        Hv = obj.hvp_f(new_params, v)
        assert torch.allclose(Hv, A.T @ (A @ v))

    def _get_mult_params_setup(self, mult_params_problem):
        """Helper to extract common setup for multiple parameter tests."""
        obj, x, Y = mult_params_problem
        p1 = x.value.shape[0]
        p2, p3 = Y.value.shape
        new_params = TensorDict({"x": -torch.ones(p1), "Y": torch.ones(p2, p3)})
        return obj, p1, p2, p3, new_params

    def test_function_evaluation_multiple_params(self, mult_params_problem):
        """Tests f_func with multiple parameters."""
        obj, p1, p2, p3, new_params = self._get_mult_params_setup(mult_params_problem)
        expected = torch.tensor(float(0.5 * p1 + p2 * p3))
        assert torch.allclose(obj.f_func(new_params), expected)

    def test_gradient_evaluation_with_multiple_params(self, mult_params_problem):
        """Test gradient computation with multiple parameters."""
        obj, p1, p2, p3, new_params = self._get_mult_params_setup(mult_params_problem)

        # Grad w.r.t x is -x
        grad_x = -torch.ones(p1)
        # Grad w.r.t. Y is 2 * Y
        grad_Y = 2 * torch.ones(p2, p3)

        # Should equal [grad_x, grad_Y]
        grads = obj.grad_f(new_params)

        # Test equality
        assert torch.allclose(grad_x, grads["x"])
        assert torch.allclose(grad_Y, grads["Y"])

    def test_hvp_with_multiple_params(self, mult_params_problem):
        """Test hvp computation with multiple parameters."""
        obj, p1, p2, p3, new_params = self._get_mult_params_setup(mult_params_problem)

        # Hessian is 60 x 60
        v = torch.ones(60)

        # Hessian w.r.t. x is I_{10}.
        # Hessian w.r.t. Y is 2 * I_{50 x 50}.
        # Flattening Y, the Hessian is 2 * 1_{50}.
        # So, the hvp with the flattened params is [1_{10}; 2 * 1_{50}].
        expected = torch.cat((torch.ones(p1), 2 * torch.ones(p2 * p3)))

        assert torch.allclose(obj.hvp_f(new_params, v), expected)


class TestOperatorSplitProx:
    """Tests for OperatorSplit proximal operator."""

    def test_prox_smooth_only_unchanged(self, smooth_only_problem):
        """Test prox returns input unchanged for smooth-only problem."""
        obj, _, _, x = smooth_only_problem
        name = obj.f.get_variable_names()[0]
        result = obj.prox(obj.f.variables_dict, 1.0)
        assert torch.allclose(result[name], x.value)

    def test_prox_lasso_matches_l1norm(self, lasso_problem):
        """Test prox corresponds to L1Norm prox for Lasso problem."""
        obj, _, _, x, r = lasso_problem
        obj_prox_result = obj.prox(obj.r.variables_dict, 1.0)["x"]
        expected = r.prox(x.value, 1.0)
        assert torch.allclose(obj_prox_result, expected)

    def test_prox_multiple_params(self):
        """Test prox is correctly computed when there are multiple parameters."""
        x = Variable(torch.tensor([0.0, 1.0, 2.0]), name="x")
        y = Variable(torch.tensor([-1.0, 5.0, -1.0]), name="y")

        f = SumSquares(x) + SumSquares(y)
        r_x = L1Norm(x)
        r_y = NonNegative(y)

        obj = OperatorSplit(f, L1Norm(x) + NonNegative(y))

        prox_x = r_x.prox(x.value, 1.0)
        prox_y = r_y.prox(y.value, 1.0)

        # Should equal [prox_x, prox_y]
        prox = obj.prox(obj.f.variables_dict, 1.0)

        # Test equality
        assert torch.allclose(prox_x, prox["x"])
        assert torch.allclose(prox_y, prox["y"])
