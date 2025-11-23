"""Test ADMMSplit class."""

import pytest
import torch

from rlaopt.atoms import L1Norm, SumSquares
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.admm_split import ADMMSplit


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
    """Tests for ADMMSplit with all smooth terms."""

    @pytest.fixture
    def single_var_smooth(self):
        """Single variable smooth problem."""
        x = Variable(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
        expr = SumSquares(x)
        obj = ADMMSplit(expr)
        return obj, x

    @pytest.fixture
    def least_squares_smooth(self, least_squares):
        """Least squares smooth problem."""
        A, b, x = least_squares
        expr = 0.5 * SumSquares(A @ x - b)
        obj = ADMMSplit(expr)
        return obj, A, b, x

    def test_initialization(self, single_var_smooth):
        """Test ADMMSplit initializes correctly with all smooth terms."""
        obj, _ = single_var_smooth
        assert obj.f is not None
        assert obj.r is not None
        assert len(obj.r) == 0
        assert obj.f.is_smooth()

    def test_A_and_b_empty(self, single_var_smooth):
        """Test A and b are appropriately sized when no decomposition."""
        obj, _ = single_var_smooth
        # A should have 0 rows (no constraints)
        assert obj.A.shape[0] == 0
        assert obj.b.shape[0] == 0

    def test_variable_values(self, single_var_smooth):
        """Test variable_values property."""
        obj, x = single_var_smooth
        var_vals = obj.variable_values
        assert var_vals.keys() == {"x"}
        assert torch.equal(var_vals["x"], x.value)

    def test_f_and_affine_variable_values(self, single_var_smooth):
        """Test f_and_affine_variable_values property."""
        obj, x = single_var_smooth
        var_vals = obj.f_and_affine_variable_values
        assert var_vals.keys() == {"x"}
        assert torch.equal(var_vals["x"], x.value)

    def test_r_variable_values_empty(self, single_var_smooth):
        """Test r_variable_values is empty when no non-smooth terms."""
        obj, _ = single_var_smooth
        r_vals = obj.r_variable_values
        assert len(r_vals.keys()) == 0

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

    def test_hvp_f_ATA_linop(self, least_squares_smooth):
        """Test hvp_f_ATA_linop with only smooth terms."""
        obj, A, _, _ = least_squares_smooth
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        rho = 1.0
        sigma = 0.1

        # Get the linear operator
        linop = obj.hvp_f_ATA_linop(new_vals, rho, sigma)

        # Test it on a vector
        v = torch.ones(p)
        result = linop @ v

        # Expected: H @ v + rho * A^T A @ v + sigma * v
        # Since A is empty for all-smooth, this is just H @ v + sigma * v
        hessian_v = A.T @ (A @ v)  # H @ v for least squares
        expected = hessian_v + sigma * v

        assert torch.allclose(result, expected)

    def test_prox_unchanged(self, single_var_smooth):
        """Test prox returns input unchanged (no non-smooth terms)."""
        obj, x = single_var_smooth
        input_vals = obj.variable_values.clone()
        result = obj.prox(input_vals, 1.0)
        assert torch.equal(result["x"], x.value)


class TestAllNonsmoothExpression:
    """Tests for ADMMSplit with all non-smooth terms."""

    @pytest.fixture
    def single_nonsmooth(self):
        """Single non-smooth term with affine expression."""
        x = Variable(torch.ones(5), name="x")
        # L1Norm is decomposable when input is affine
        expr = L1Norm(x)
        obj = ADMMSplit(expr)
        return obj, x

    @pytest.fixture
    def lasso_constraint(self):
        """Lasso with affine constraint."""
        x = Variable(torch.ones(10), name="x")
        A = torch.randn(5, 10)
        b = torch.randn(5)
        expr = L1Norm(A @ x - b)
        obj = ADMMSplit(expr)
        return obj, x, A, b

    def test_initialization(self, single_nonsmooth):
        """Test initialization with all non-smooth terms."""
        obj, _ = single_nonsmooth
        assert obj.f is not None
        assert obj.r is not None
        assert len(obj.r) == 1

    def test_A_shape(self, single_nonsmooth):
        """Test A has correct shape."""
        obj, x = single_nonsmooth
        # A should map from x (5-dim) to auxiliary variable (5-dim)
        assert obj.A.shape == (5, 5)

    def test_A_is_identity(self, single_nonsmooth):
        """Test A is identity when input is just a variable."""
        obj, _ = single_nonsmooth
        # For L1Norm(x), decomposition is L1Norm(z) with z = x
        # So A should be identity
        v = torch.randn(5)
        result = obj.A @ v
        assert torch.allclose(result, v)

    def test_b_is_zero(self, single_nonsmooth):
        """Test b is zero when affine expr has no bias."""
        obj, _ = single_nonsmooth
        assert torch.allclose(obj.b, torch.zeros(5))

    def test_A_matches_affine_expr(self, lasso_constraint):
        """Test A matches the matrix in affine expression."""
        obj, x, A_matrix, b_vec = lasso_constraint
        # The decomposition is L1Norm(z) with z = A @ x - b
        # So obj.A should match A_matrix
        v = torch.randn(10)
        result = obj.A @ v
        expected = A_matrix @ v
        assert torch.allclose(result, expected)

    def test_b_matches_affine_expr(self, lasso_constraint):
        """Test b matches the bias in affine expression."""
        obj, x, A_matrix, b_vec = lasso_constraint
        # Note: ADMMSplit negates b internally
        assert torch.allclose(obj.b, b_vec)

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

    def test_hvp_f_ATA_linop(self, lasso_constraint):
        """Test hvp_f_ATA_linop with only non-smooth terms."""
        obj, x, A_matrix, _ = lasso_constraint
        new_vals = TensorDict({"x": torch.ones(10)})
        rho = 2.0
        sigma = 0.5

        linop = obj.hvp_f_ATA_linop(new_vals, rho, sigma)

        # Test on a vector
        v = torch.randn(10)
        result = linop @ v

        # Expected: rho * A^T A @ v + sigma * v (no Hessian for all-nonsmooth)
        expected = rho * (A_matrix.T @ (A_matrix @ v)) + sigma * v
        assert torch.allclose(result, expected)

    def test_r_variable_values(self, single_nonsmooth):
        """Test r_variable_values contains auxiliary variables."""
        obj, _ = single_nonsmooth
        r_vals = obj.r_variable_values
        # Should have auxiliary variables from decomposition
        assert len(r_vals.keys()) > 0


class TestMixedSameVariable:
    """Tests for ADMMSplit with mixed smooth/non-smooth on same variable."""

    @pytest.fixture
    def elastic_net_problem(self, least_squares):
        """Elastic net: smooth loss + decomposable L1."""
        A, b, x = least_squares
        f = 0.5 * SumSquares(A @ x - b)
        r = L1Norm(x)
        expr = f + r
        obj = ADMMSplit(expr)
        return obj, A, b, x

    def test_initialization(self, elastic_net_problem):
        """Test initialization with mixed smooth/non-smooth on same variable."""
        obj, _, _, _ = elastic_net_problem
        assert obj.f is not None
        assert obj.r is not None
        assert len(obj.r) == 1
        assert obj.f.is_smooth()

    def test_A_shape(self, elastic_net_problem):
        """Test A has correct shape."""
        obj, _, _, x = elastic_net_problem
        # A maps from x to auxiliary variable z (same dimension)
        assert obj.A.shape == (16, 16)

    def test_A_is_identity(self, elastic_net_problem):
        """Test A is identity for L1Norm(x)."""
        obj, _, _, _ = elastic_net_problem
        p = obj.A.shape[1]
        v = torch.randn(p)
        result = obj.A @ v
        assert torch.allclose(result, v)

    def test_func_f(self, elastic_net_problem):
        """Test func_f method."""
        obj, A, b, _ = elastic_net_problem
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        result = obj.func_f(new_vals)
        expected = 0.5 * torch.linalg.norm(b) ** 2
        assert torch.allclose(result, expected)

    def test_grad_f(self, elastic_net_problem):
        """Test grad_f method."""
        obj, A, b, _ = elastic_net_problem
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        grads = obj.grad_f(new_vals)
        assert torch.allclose(grads["x"], -A.T @ b)

    def test_hvp_f(self, elastic_net_problem):
        """Test hvp_f method."""
        obj, A, _, _ = elastic_net_problem
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        v = torch.ones(p)
        result = obj.hvp_f(new_vals, v)
        assert torch.allclose(result, A.T @ (A @ v))

    def test_hvp_f_ATA_linop(self, elastic_net_problem):
        """Test hvp_f_ATA_linop combines terms correctly."""
        obj, A, _, _ = elastic_net_problem
        p = A.shape[1]
        new_vals = TensorDict({"x": torch.zeros(p)})
        rho = 1.5
        sigma = 0.2

        linop = obj.hvp_f_ATA_linop(new_vals, rho, sigma)
        v = torch.randn(p)
        result = linop @ v

        # Expected: H @ v + rho * I @ v + sigma * v
        # (A is identity for this decomposition)
        hessian_v = A.T @ (A @ v)
        expected = hessian_v + rho * v + sigma * v
        assert torch.allclose(result, expected)


class TestMixedDisjointVariables:
    """Tests for ADMMSplit with mixed smooth/non-smooth on disjoint variables."""

    @pytest.fixture
    def smooth_has_extra_var(self):
        """Problem with smooth on x, non-smooth on y."""
        x = Variable(torch.ones(5), name="x")
        y = Variable(torch.zeros(3), name="y")
        expr = SumSquares(x) + L1Norm(y)
        obj = ADMMSplit(expr)
        return obj, x, y

    @pytest.fixture
    def nonsmooth_has_extra_var(self):
        """Problem where affine expr references variable not in smooth."""
        x = Variable(torch.ones(5), name="x")
        y = Variable(torch.zeros(3), name="y")
        A = torch.randn(3, 5)
        # Affine expression: A @ x + y
        expr = SumSquares(x) + L1Norm(A @ x + y)
        obj = ADMMSplit(expr)
        return obj, x, y, A

    def test_initialization_smooth_extra(self, smooth_has_extra_var):
        """Test initialization with smooth having extra variable."""
        obj, _, _ = smooth_has_extra_var
        assert obj.f is not None
        assert obj.r is not None
        assert len(obj.r) == 1

    def test_f_and_affine_variable_values_smooth_extra(self, smooth_has_extra_var):
        """Test f_and_affine_variable_values includes both variables."""
        obj, x, y = smooth_has_extra_var
        vals = obj.f_and_affine_variable_values
        # Should include both x (from f) and y (from affine expr)
        assert "x" in vals
        assert "y" in vals

    def test_A_shape_smooth_extra(self, smooth_has_extra_var):
        """Test A shape when smooth has extra variable."""
        obj, _, _ = smooth_has_extra_var
        # A maps from (x, y) -> z where z has dimension of y (3)
        # Input dimension is 5 + 3 = 8
        assert obj.A.shape[1] == 8
        assert obj.A.shape[0] == 3

    def test_A_selects_y_smooth_extra(self, smooth_has_extra_var):
        """Test A correctly selects y variable."""
        obj, _, _ = smooth_has_extra_var
        # A should select the y part: [0, 0, 0, 0, 0, 1, 0, 0]^T etc.
        v = torch.cat([torch.ones(5), torch.tensor([1.0, 2.0, 3.0])])
        result = obj.A @ v
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result, expected)

    def test_initialization_nonsmooth_extra(self, nonsmooth_has_extra_var):
        """Test initialization when affine expr has extra variable."""
        obj, _, _, _ = nonsmooth_has_extra_var
        assert obj.f is not None
        assert obj.r is not None
        assert len(obj.r) == 1

    def test_f_and_affine_variable_values_nonsmooth_extra(
        self, nonsmooth_has_extra_var
    ):
        """Test f_and_affine_variable_values includes affine variables."""
        obj, x, y, _ = nonsmooth_has_extra_var
        vals = obj.f_and_affine_variable_values
        # Should include both x (from f and affine) and y (from affine)
        assert "x" in vals
        assert "y" in vals

    def test_A_shape_nonsmooth_extra(self, nonsmooth_has_extra_var):
        """Test A shape when affine expr has extra variable."""
        obj, _, _, A_matrix = nonsmooth_has_extra_var
        # Input: (x, y) = 5 + 3 = 8
        # Output: z has dimension 3
        assert obj.A.shape == (3, 8)

    def test_A_computes_affine_expr(self, nonsmooth_has_extra_var):
        """Test A correctly computes A @ x + y."""
        obj, _, _, A_matrix = nonsmooth_has_extra_var
        x_val = torch.randn(5)
        y_val = torch.randn(3)
        v = torch.cat([x_val, y_val])
        result = obj.A @ v

        expected = A_matrix @ x_val + y_val
        assert torch.allclose(result, expected)

    def test_grad_f_nonsmooth_extra(self, nonsmooth_has_extra_var):
        """Test grad_f only affects smooth variables."""
        obj, x, y, _ = nonsmooth_has_extra_var
        new_vals = TensorDict({"x": torch.ones(5), "y": torch.zeros(3)})
        grads = obj.grad_f(new_vals)

        # x is in smooth part, should have non-zero gradient
        assert torch.allclose(grads["x"], 2 * torch.ones(5))
        # y is not in smooth part, should be zero
        assert torch.allclose(grads["y"], torch.zeros(3))


class TestMultipleDecomposableAtoms:
    """Tests for ADMMSplit with multiple decomposable atoms."""

    @pytest.fixture
    def multiple_atoms(self):
        """Problem with multiple decomposable atoms."""
        x = Variable(torch.ones(5), name="x")
        y = Variable(torch.zeros(3), name="y")
        # Two decomposable atoms
        expr = L1Norm(x) + L1Norm(y)
        obj = ADMMSplit(expr)
        return obj, x, y

    def test_initialization(self, multiple_atoms):
        """Test initialization with multiple atoms."""
        obj, _, _ = multiple_atoms
        assert obj.r is not None
        assert len(obj.r) == 2

    def test_A_stacks_correctly(self, multiple_atoms):
        """Test A stacks multiple decompositions."""
        obj, _, _ = multiple_atoms
        # A should stack: first 5 rows for x, next 3 rows for y
        # Total rows: 5 + 3 = 8
        # Total cols: 5 + 3 = 8
        assert obj.A.shape == (8, 8)

    def test_A_is_identity(self, multiple_atoms):
        """Test A is identity when both are simple decompositions."""
        obj, _, _ = multiple_atoms
        v = torch.randn(8)
        result = obj.A @ v
        assert torch.allclose(result, v)

    def test_b_stacks_correctly(self, multiple_atoms):
        """Test b stacks correctly."""
        obj, _, _ = multiple_atoms
        assert obj.b.shape[0] == 8
        assert torch.allclose(obj.b, torch.zeros(8))


class TestErrorConditions:
    """Tests for error conditions in ADMMSplit."""

    def test_non_1d_variable_raises_error(self):
        """Test that non-1D variables raise ValueError."""
        X = Variable((3, 4), name="X")  # 2D variable
        expr = L1Norm(X)

        with pytest.raises(ValueError, match="1-dimensional"):
            ADMMSplit(expr)

    def test_non_decomposable_atom_raises_error(self):
        """Test that non-decomposable atom raises ValueError."""
        x = Variable(torch.ones(5), name="x")
        # SumSquares doesn't implement decompose (returns None)
        expr = SumSquares(x) + SumSquares(x)

        with pytest.raises(ValueError, match="decomposable"):
            ADMMSplit(expr)

    def test_non_atom_nonsmooth_raises_error(self):
        """Test that non-Atom non-smooth term raises ValueError."""
        x = Variable(torch.ones(5), name="x")
        # x ** 2 is not an Atom
        non_atom = x**2
        expr = SumSquares(x) + non_atom

        with pytest.raises(ValueError, match="Atom"):
            ADMMSplit(expr)
