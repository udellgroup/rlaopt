"""Tests for Affine atom."""

import pytest
import torch

from rlaopt.atoms.affine import Affine
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    x = Variable((5,), name="x")
    x.value.data = torch.ones(5)
    return x


@pytest.fixture
def matrix_var():
    """Create a matrix variable."""
    X = Variable((4, 3), name="X")
    X.value.data = torch.ones(4, 3)
    return X


@pytest.fixture
def simple_affine_data():
    """Create simple affine transformation data."""
    return {"A": torch.eye(5), "b": torch.zeros(5)}


@pytest.fixture
def nontrivial_affine_data():
    """Create non-trivial affine transformation data."""
    return {
        "A": torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 3.0],
            ]
        ),
        "b": torch.tensor([1.0, -1.0, 2.0]),
    }


class TestAffine:
    """Test Affine transformation atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_variable(self, vector_var, simple_affine_data):
        """Test initialization with Variable input."""
        affine = Affine(vector_var, **simple_affine_data)
        assert affine.A is not None
        assert affine.b is not None

    def test_init_with_expression_input(self, vector_var, simple_affine_data):
        """Test initialization with Expression input."""
        # Create an expression from a variable
        expr = vector_var + 1.0
        affine = Affine(expr, **simple_affine_data)
        assert affine.A is not None
        assert affine.b is not None

    def test_init_stores_transformation_matrices(
        self, vector_var, nontrivial_affine_data
    ):
        """Test initialization stores A and b correctly."""
        affine = Affine(vector_var, **nontrivial_affine_data)
        assert torch.equal(affine.get_buffer("A"), nontrivial_affine_data["A"])
        assert torch.equal(affine.get_buffer("b"), nontrivial_affine_data["b"])

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_with_identity_transformation(self, vector_var, simple_affine_data):
        """Test forward() with identity transformation (A=I, b=0)."""
        affine = Affine(vector_var, **simple_affine_data)
        result = affine.forward()
        expected = torch.ones(5)
        assert torch.allclose(result, expected)

    def test_forward_computes_affine_transformation(
        self, vector_var, nontrivial_affine_data
    ):
        """Test forward() correctly computes A @ x + b."""
        affine = Affine(vector_var, **nontrivial_affine_data)
        result = affine.forward()

        # Manual computation: A @ ones(5) + b
        # Row 0: [1,2,0,0,0] @ [1,1,1,1,1] + 1 = 3 + 1 = 4
        # Row 1: [0,1,1,0,0] @ [1,1,1,1,1] + (-1) = 2 - 1 = 1
        # Row 2: [0,0,1,2,3] @ [1,1,1,1,1] + 2 = 6 + 2 = 8
        expected = torch.tensor([4.0, 1.0, 8.0])
        assert torch.allclose(result, expected)

    def test_forward_with_expression_input(self, vector_var):
        """Test forward() with Expression input evaluates the expression first."""
        # Create expression: x + 2
        expr = vector_var + 2.0
        A = torch.eye(5) * 2
        b = torch.ones(5)
        affine = Affine(expr, A, b)

        # vector_var.value = ones(5), so expr = ones(5) + 2 = 3*ones(5)
        # Result: 2*I @ 3*ones(5) + ones(5) = 6*ones(5) + ones(5) = 7*ones(5)
        result = affine.forward()
        expected = torch.ones(5) * 7
        assert torch.allclose(result, expected)

    def test_forward_with_matrix_input(self, matrix_var):
        """Test forward() works with matrix inputs."""
        # Flatten transformation: sum rows
        A = torch.ones(3, 4)  # Each row sums the 4 columns
        b = torch.zeros(3)
        affine = Affine(matrix_var, A, b)

        # matrix_var is 4x3 of ones, A is 3x4, so A @ X gives 3x3
        # Each element = sum of row of A @ column of X = 4 * 1 = 4
        result = affine.forward()
        assert result.shape == (3, 3)

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_true(self, vector_var, simple_affine_data):
        """Test affine transformation is smooth when input is smooth."""
        affine = Affine(vector_var, **simple_affine_data)
        assert affine.is_smooth() is True

    def test_is_smooth_returns_false_with_nonsmooth_input(self, vector_var):
        """Test is_smooth() returns False when input expression is non-smooth."""

        # Create a mock non-smooth expression
        class NonSmoothExpr(Expression):
            def __init__(self):
                super().__init__()

            def is_smooth(self):
                return False

            def is_proxable(self):
                return False

            def forward(self):
                return torch.ones(5)

            def tree(self):
                raise NotImplementedError()

        nonsmooth = NonSmoothExpr()
        A = torch.eye(5)
        b = torch.zeros(5)
        affine = Affine(nonsmooth, A, b)
        assert affine.is_smooth() is False

    def test_is_proxable_returns_false(self, vector_var, simple_affine_data):
        """Test affine transformation is not proxable."""
        affine = Affine(vector_var, **simple_affine_data)
        assert affine.is_proxable() is False

    def test_prox_raises_not_implemented(self, vector_var, simple_affine_data):
        """Test prox() raises NotImplementedError."""
        affine = Affine(vector_var, **simple_affine_data)
        with pytest.raises(NotImplementedError, match="not proxable"):
            affine.prox(TensorDict({vector_var.name: torch.ones(5)}), 1.0)

    # ----------------------
    # Edge cases
    # ----------------------

    def test_with_zero_matrix(self, vector_var):
        """Test affine with zero transformation matrix."""
        A = torch.zeros(3, 5)
        b = torch.ones(3) * 5
        affine = Affine(vector_var, A, b)
        result = affine.forward()
        assert torch.allclose(result, b)  # Should return just b

    def test_with_zero_bias(self, vector_var):
        """Test affine with zero bias vector."""
        A = torch.eye(5) * 2
        b = torch.zeros(5)
        affine = Affine(vector_var, A, b)
        result = affine.forward()
        expected = torch.ones(5) * 2  # Just A @ x
        assert torch.allclose(result, expected)

    def test_gradient_flows_through_transformation(
        self, vector_var, simple_affine_data
    ):
        """Test gradients flow through affine transformation."""
        vector_var.value.requires_grad_(True)
        affine = Affine(vector_var, **simple_affine_data)
        result = affine.forward()
        loss = result.sum()
        loss.backward()

        assert vector_var.value.grad is not None
