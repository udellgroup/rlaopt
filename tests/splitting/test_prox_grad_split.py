"""Test ProxGradSplit class."""

import pytest
import torch

from rlaopt.atoms import L1Norm, NonNegative, SumSquares
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.prox_grad_split import ProxGradSplit


@pytest.fixture
def least_squares():
    """Least squares problem fixture reused across tests."""
    n, p = 32, 16
    torch.manual_seed(0)
    A = torch.randn(n, p, dtype=torch.float32) / (n**0.5)
    b = torch.randn(n, dtype=torch.float32) / (n**0.5)
    x = Variable(torch.ones(p, dtype=torch.float32), name="x")
    return A, b, x


class TestAllSmoothExpression:
    """Tests for ProxGradSplit with all smooth terms."""

    @pytest.fixture
    def single_var_smooth(self):
        """Single variable smooth problem."""
        x = Variable(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
        expr = SumSquares(x)
        obj = ProxGradSplit(expr)
        return obj, x

    @pytest.fixture
    def least_squares_smooth(self, least_squares):
        """Least squares smooth problem."""
        A, b, x = least_squares
        expr = 0.5 * SumSquares(A @ x - b)
        obj = ProxGradSplit(expr)
        return obj, A, b, x

    def test_initialization(self, single_var_smooth):
        """Test ProxGradSplit initializes correctly with all smooth terms."""
        obj, _ = single_var_smooth
        assert obj._f is not None
        assert obj._r is not None
        assert len(obj._r) == 0
        assert obj.f.is_smooth()

    def test_has_non_smooth_component(self, single_var_smooth):
        """has_non_smooth_component is False when objective is all smooth."""
        obj, _ = single_var_smooth
        assert obj.has_non_smooth_component is False

    def test_variable_values(self, single_var_smooth):
        """Test variable_values property."""
        obj, x = single_var_smooth
        var_vals = obj.variable_values
        assert var_vals.keys() == {"x"}
        assert torch.equal(var_vals["x"], x.value)

    def test_evaluate(self, single_var_smooth):
        """Test evaluate method."""
        obj, x = single_var_smooth
        new_vals = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])})
        result = obj.evaluate(new_vals)
        expected = torch.tensor(55.0)  # 1+4+9+16+25
        assert torch.allclose(result, expected)

    def test_func_f(self, least_squares_smooth):
        """Test func_f method."""
        obj, A, b, _ = least_squares_smooth
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        result = obj.func_f(new_vals)
        expected = 0.5 * torch.linalg.norm(b) ** 2
        assert torch.allclose(result, expected)

    def test_grad_f(self, least_squares_smooth):
        """Test grad_f method."""
        obj, A, b, _ = least_squares_smooth
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        grads = obj.grad_f(new_vals)
        assert "x" in grads
        assert torch.allclose(grads["x"], -A.T @ b)

    def test_hvp_f(self, least_squares_smooth):
        """Test hvp_f method."""
        obj, A, _, _ = least_squares_smooth
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        v = torch.ones(p)
        result = obj.hvp_f(new_vals, v)
        expected = A.T @ (A @ v)
        assert torch.allclose(result, expected)

    def test_prox_unchanged(self, single_var_smooth):
        """Test prox returns input unchanged (no non-smooth terms)."""
        obj, x = single_var_smooth
        input_vals = obj.variable_values.clone()
        result = obj.prox(input_vals, 1.0)
        assert torch.equal(result["x"], x.value)


class TestAllNonsmoothExpression:
    """Tests for ProxGradSplit with all non-smooth terms."""

    @pytest.fixture
    def single_nonsmooth(self):
        """Single non-smooth term."""
        x = Variable(torch.ones(5), name="x")
        expr = L1Norm(x)
        obj = ProxGradSplit(expr)
        return obj, x

    def test_initialization(self, single_nonsmooth):
        """Test initialization with all non-smooth terms."""
        obj, _ = single_nonsmooth
        assert obj._f is not None
        assert obj._r is not None
        assert len(obj._r) == 1

    def test_has_non_smooth_component(self, single_nonsmooth):
        """has_non_smooth_component is True when objective is purely non-smooth."""
        obj, _ = single_nonsmooth
        assert obj.has_non_smooth_component is True

    def test_evaluate(self, single_nonsmooth):
        """Test evaluate method."""
        obj, _ = single_nonsmooth
        new_vals = TensorDict({"x": torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])})
        result = obj.evaluate(new_vals)
        expected = torch.tensor(15.0)  # L1 norm
        assert torch.allclose(result, expected)

    def test_func_f_returns_zero(self, single_nonsmooth):
        """Test func_f returns 0 when there's no smooth part."""
        obj, _ = single_nonsmooth
        new_vals = TensorDict({"x": torch.ones(5)})
        result = obj.func_f(new_vals)
        assert torch.allclose(result, torch.tensor(0.0))

    def test_grad_f_returns_zeros(self, single_nonsmooth):
        """Test grad_f returns zeros when there's no smooth part."""
        obj, _ = single_nonsmooth
        new_vals = TensorDict({"x": torch.ones(5)})
        grads = obj.grad_f(new_vals)
        assert torch.allclose(grads["x"], torch.zeros(5))

    def test_hvp_f_returns_zeros(self, single_nonsmooth):
        """Test hvp_f returns zeros when there's no smooth part."""
        obj, _ = single_nonsmooth
        new_vals = TensorDict({"x": torch.ones(5)})
        v = torch.ones(5)
        result = obj.hvp_f(new_vals, v)
        assert torch.allclose(result, torch.zeros(5))

    def test_prox(self, single_nonsmooth):
        """Test prox method."""
        obj, _ = single_nonsmooth
        input_vals = TensorDict({"x": torch.tensor([-2.0, 3.0, -1.0, 0.5, -4.0])})
        result = obj.prox(input_vals, 1.0)
        expected = torch.tensor([-1.0, 2.0, 0.0, 0.0, -3.0])  # Soft thresholding
        assert torch.allclose(result["x"], expected)


class TestMixedSameVariable:
    """Tests for ProxGradSplit with mixed smooth/non-smooth on same variable."""

    @pytest.fixture
    def lasso_problem(self, least_squares):
        """Lasso problem: smooth and non-smooth on same variable."""
        A, b, x = least_squares
        f = 0.5 * SumSquares(A @ x - b)
        r = L1Norm(x)
        expr = f + r
        obj = ProxGradSplit(expr)
        return obj, A, b, x, r

    def test_initialization(self, lasso_problem):
        """Test initialization with mixed smooth/non-smooth on same variable."""
        obj, _, _, _, _ = lasso_problem
        assert obj._f is not None
        assert obj._r is not None
        assert len(obj._r) == 1
        assert obj.f.is_smooth()
        assert obj.r[0].is_proxable()
        # Both should use variable x
        assert "x" in obj.f.get_variable_names()
        assert "x" in obj.r[0].get_variable_names()

    def test_has_non_smooth_component(self, lasso_problem):
        """has_non_smooth_component is True for a mixed smooth/non-smooth objective."""
        obj, _, _, _, _ = lasso_problem
        assert obj.has_non_smooth_component is True

    def test_evaluate(self, lasso_problem):
        """Test evaluate method."""
        obj, A, b, _, _ = lasso_problem
        p = A.shape[1]
        test_vals = torch.tensor([1.0, -2.0, 3.0] + [0.0] * (p - 3))
        new_vals = TensorDict({"x": test_vals})

        # Expected: f + r
        f_val = 0.5 * torch.sum((A @ test_vals - b) ** 2)
        r_val = torch.sum(torch.abs(test_vals))
        expected = f_val + r_val

        result = obj.evaluate(new_vals)
        assert torch.allclose(result, expected)

    def test_func_f(self, lasso_problem):
        """Test func_f method."""
        obj, A, b, _, _ = lasso_problem
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        result = obj.func_f(new_vals)
        expected = 0.5 * torch.linalg.norm(b) ** 2
        assert torch.allclose(result, expected)

    def test_grad_f(self, lasso_problem):
        """Test grad_f method."""
        obj, A, b, _, _ = lasso_problem
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        grads = obj.grad_f(new_vals)
        assert torch.allclose(grads["x"], -A.T @ b)

    def test_hvp_f(self, lasso_problem):
        """Test hvp_f method."""
        obj, A, _, _, _ = lasso_problem
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        v = torch.ones(p)
        result = obj.hvp_f(new_vals, v)
        assert torch.allclose(result, A.T @ (A @ v))

    def test_prox(self, lasso_problem):
        """Test prox method."""
        obj, _, _, x, r = lasso_problem
        input_vals = obj.variable_values.clone()
        result = obj.prox(input_vals, 1.0)
        expected = r.prox(TensorDict({"x": x.value}), 1.0)["x"]
        assert torch.allclose(result["x"], expected)


class TestMixedDisjointVariables:
    """Tests for ProxGradSplit with mixed smooth/non-smooth on disjoint variables."""

    @pytest.fixture
    def disjoint_problem(self):
        """Problem with smooth on x, non-smooth on y."""
        x = Variable(torch.ones(5), name="x")
        y = Variable(torch.zeros(3), name="y")
        expr = SumSquares(x) + L1Norm(y)
        obj = ProxGradSplit(expr)
        return obj, x, y

    def test_initialization(self, disjoint_problem):
        """Test initialization with disjoint variables."""
        obj, _, _ = disjoint_problem
        assert obj._f is not None
        assert obj._r is not None
        assert len(obj._r) == 1

        f_vars = set(obj.f.get_variable_names())
        r_vars = set(obj.r[0].get_variable_names())

        assert f_vars == {"x"}
        assert r_vars == {"y"}

    def test_evaluate(self, disjoint_problem):
        """Test evaluate method."""
        obj, _, _ = disjoint_problem
        new_vals = TensorDict(
            {
                "x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
                "y": torch.tensor([-1.0, 2.0, -3.0]),
            }
        )
        result = obj.evaluate(new_vals)
        # Expected: sum(x^2) + sum(|y|) = 55 + 6 = 61
        expected = torch.tensor(61.0)
        assert torch.allclose(result, expected)

    def test_func_f(self, disjoint_problem):
        """Test func_f method."""
        obj, _, _ = disjoint_problem
        new_vals = TensorDict(
            {"x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), "y": torch.zeros(3)}
        )
        result = obj.func_f(new_vals)
        expected = torch.tensor(55.0)  # Only smooth term
        assert torch.allclose(result, expected)

    def test_grad_f(self, disjoint_problem):
        """Test grad_f method."""
        obj, _, _ = disjoint_problem
        new_vals = TensorDict({"x": torch.ones(5), "y": torch.zeros(3)})
        grads = obj.grad_f(new_vals)
        assert torch.allclose(grads["x"], 2 * torch.ones(5))
        # y is not in smooth part, but grad should still be present
        assert "y" in grads

    def test_hvp_f(self, disjoint_problem):
        """Test hvp_f method."""
        obj, _, _ = disjoint_problem
        new_vals = TensorDict({"x": torch.zeros(5), "y": torch.zeros(3)})
        v = torch.ones(8)  # 5 + 3
        result = obj.hvp_f(new_vals, v)
        # Hessian is 2*I for x, 0 for y
        expected = torch.cat([2 * torch.ones(5), torch.zeros(3)])
        assert torch.allclose(result, expected)

    def test_prox(self, disjoint_problem):
        """Test prox method."""
        obj, x, y = disjoint_problem
        input_vals = TensorDict(
            {
                "x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
                "y": torch.tensor([-2.0, 3.0, -1.0]),
            }
        )
        result = obj.prox(input_vals, 1.0)

        # x should be unchanged (not in non-smooth)
        assert torch.equal(result["x"], input_vals["x"])
        # y should be soft-thresholded
        expected_y = torch.tensor([-1.0, 2.0, 0.0])
        assert torch.allclose(result["y"], expected_y)


class TestMixedOverlappingVariables:
    """Tests for ProxGradSplit with mixed smooth/non-smooth on overlapping variables."""

    @pytest.fixture
    def overlapping_problem(self):
        """Problem with overlapping variables in smooth and non-smooth."""
        x = Variable(torch.ones(5), name="x")
        y = Variable(torch.zeros(3), name="y")
        z = Variable(torch.ones(4), name="z")
        # Smooth: x, y; Non-smooth: y, z
        expr = SumSquares(x) + SumSquares(y) + NonNegative(y) + L1Norm(z)
        obj = ProxGradSplit(expr)
        return obj, x, y, z

    def test_initialization(self, overlapping_problem):
        """Test initialization with overlapping variables."""
        obj, _, _, _ = overlapping_problem
        assert obj._f is not None
        assert obj._r is not None
        assert len(obj._r) == 2

        f_vars = set(obj.f.get_variable_names())
        r_vars = set()
        for r_atom in obj.r:
            r_vars.update(r_atom.get_variable_names())

        assert f_vars == {"x", "y"}
        assert r_vars == {"y", "z"}

    def test_evaluate(self, overlapping_problem):
        """Test evaluate method."""
        obj, _, _, _ = overlapping_problem
        new_vals = TensorDict(
            {
                "x": torch.ones(5),
                "y": torch.tensor([1.0, 2.0, 3.0]),
                "z": torch.tensor([-1.0, 2.0, -3.0, 4.0]),
            }
        )
        result = obj.evaluate(new_vals)
        # sum(x^2) + sum(y^2) + indicator(y>=0) + sum(|z|)
        # = 5 + 14 + 0 + 10 = 29
        expected = torch.tensor(29.0)
        assert torch.allclose(result, expected)

    def test_func_f(self, overlapping_problem):
        """Test func_f method."""
        obj, _, _, _ = overlapping_problem
        new_vals = TensorDict(
            {
                "x": torch.ones(5),
                "y": torch.tensor([1.0, 2.0, 3.0]),
                "z": torch.zeros(4),
            }
        )
        result = obj.func_f(new_vals)
        # Only smooth: sum(x^2) + sum(y^2) = 5 + 14 = 19
        expected = torch.tensor(19.0)
        assert torch.allclose(result, expected)

    def test_grad_f(self, overlapping_problem):
        """Test grad_f method."""
        obj, _, _, _ = overlapping_problem
        new_vals = TensorDict(
            {
                "x": torch.ones(5),
                "y": torch.tensor([1.0, 2.0, 3.0]),
                "z": torch.zeros(4),
            }
        )
        grads = obj.grad_f(new_vals)
        assert torch.allclose(grads["x"], 2 * torch.ones(5))
        assert torch.allclose(grads["y"], 2 * torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(grads["z"], torch.zeros(4))

    def test_prox(self, overlapping_problem):
        """Test prox method."""
        obj, _, _, _ = overlapping_problem
        input_vals = TensorDict(
            {
                "x": torch.ones(5),
                "y": torch.tensor([1.0, -0.5, 2.0]),
                "z": torch.tensor([-2.0, 3.0, -1.0, 0.5]),
            }
        )
        result = obj.prox(input_vals, 1.0)

        # x should be unchanged
        assert torch.equal(result["x"], input_vals["x"])
        # y should be clamped to non-negative
        expected_y = torch.tensor([1.0, 0.0, 2.0])
        assert torch.allclose(result["y"], expected_y)
        # z should be soft-thresholded
        expected_z = torch.tensor([-1.0, 2.0, 0.0, 0.0])
        assert torch.allclose(result["z"], expected_z)


class TestErrorConditions:
    """Tests for error conditions in _attempt_split."""

    def test_non_proxable_nonsmooth_raises_error(self):
        """Test that non-proxable non-smooth term raises ValueError."""
        x = Variable(torch.ones(5), name="x")
        f = SumSquares(x)
        non_proxable = L1Norm(x) ** 2
        expr = f + non_proxable

        with pytest.raises(ValueError, match="proxable atoms"):
            ProxGradSplit(expr)

    def test_overlapping_nonsmooth_variables_raises_error(self):
        """Test that overlapping variables in non-smooth terms raises error."""
        x = Variable(torch.ones(5), name="x")
        y = Variable(torch.zeros(5), name="y")
        expr = SumSquares(y) + L1Norm(x) + NonNegative(x)

        with pytest.raises(ValueError, match="disjoint sets of variables"):
            ProxGradSplit(expr)
