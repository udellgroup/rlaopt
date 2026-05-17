"""Tests for Variable class."""

import pytest
import torch

from rlaopt.expression import ExprTree
from rlaopt.expression.variable import Variable


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((5,), name="vector")


@pytest.fixture
def matrix_var():
    """Create a matrix variable."""
    return Variable((3, 4), name="matrix")


class TestVariable:
    """Test Variable concrete implementation."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_tuple_and_single_int(self):
        """Test initialization with tuple and single int both work."""
        tuple_var = Variable((5,), name="tuple")
        int_var = Variable(5, name="int")
        assert tuple_var.value.shape == int_var.value.shape == torch.Size([5])
        assert torch.equal(tuple_var.value, torch.zeros(5))

    def test_init_with_tensor(self):
        """Test initialization with existing tensor."""
        data = torch.randn(3, 3)
        var = Variable(data, name="from_tensor")
        assert torch.equal(var.value, data)

    def test_init_with_custom_parameters(self):
        """Test initialization with custom dtype, device, and requires_grad."""
        var = Variable(
            5,
            dtype=torch.float64,
            device=torch.device("cpu"),
            requires_grad=False,
            name="custom",
        )
        assert var.value.dtype == torch.float64
        assert var.value.device.type == "cpu"
        assert var.value.requires_grad is False
        assert len(list(var.parameters())) == 1

    def test_init_with_invalid_type_raises_error(self):
        """Test initialization with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="size must be"):
            Variable([5, 10], name="invalid")

    # ----------------------
    # Classmethod tests
    # ----------------------

    def test_like_creates_variable_with_same_shape(self, vector_var):
        """Test Variable.like() creates variable with same shape as expression."""
        new_var = Variable.like(vector_var)
        assert new_var.value.shape == vector_var.value.shape

    def test_like_creates_variable_with_same_dtype(self):
        """Test Variable.like() creates variable with same dtype as expression."""
        original = Variable((3, 3), dtype=torch.float64, name="original")
        new_var = Variable.like(original)
        assert new_var.value.dtype == original.value.dtype

    def test_like_creates_variable_with_same_device(self):
        """Test Variable.like() creates variable with same device as expression."""
        original = Variable((3, 3), device=torch.device("cpu"), name="original")
        new_var = Variable.like(original)
        assert new_var.value.device == original.value.device

    def test_like_works_with_expression(self):
        """Test Variable.like() works with any expression, not just Variable."""
        x = Variable((5,), name="x", dtype=torch.float64)
        y = Variable((5,), name="y", dtype=torch.float64)
        new_var = Variable.like(x + y)
        assert new_var.value.shape == torch.Size([5])
        assert new_var.value.dtype == torch.float64

    # ----------------------
    # ID and name tests
    # ----------------------

    def test_name_and_id_management(self):
        """Test custom and auto-generated names and IDs."""
        custom_var = Variable((5,), var_id=42, name="custom")
        auto_var = Variable((5,))

        assert custom_var.name == "custom"
        assert custom_var.id == 42
        assert auto_var.name.startswith("var_")
        assert custom_var.id != auto_var.id

    def test_init_with_invalid_name_raises_error(self):
        """Test initialization with non-string name raises TypeError."""
        with pytest.raises(TypeError, match="Expected name to be a string"):
            Variable((5,), name=123)

    # ----------------------
    # Property tests
    # ----------------------

    def test_properties_return_expected_values(self, vector_var):
        """Test id, name, and value properties."""
        assert isinstance(vector_var.id, int)
        assert vector_var.name == "vector"
        assert isinstance(vector_var.value, torch.nn.Parameter)

    def test_value_setter_updates_parameter(self, vector_var):
        """Test value setter replaces parameter and preserves requires_grad."""
        original_requires_grad = vector_var.value.requires_grad
        new_value = torch.ones(5)
        vector_var.value = new_value
        assert torch.equal(vector_var.value, new_value)
        assert vector_var.value.requires_grad == original_requires_grad

    # ----------------------
    # Abstract method implementations
    # ----------------------

    def test_is_smooth(self, vector_var):
        """Test variables are smooth."""
        assert vector_var.is_smooth() is True

    def test_tree(self, vector_var):
        """Test tree() returns correct structure."""
        assert vector_var.tree() == ExprTree("Variable(vector)")

    # ----------------------
    # is_affine tests
    # ----------------------

    def test_is_affine(self, vector_var):
        """Test variables are affine."""
        assert vector_var.is_affine() is True

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_returns_parameter_value(self, vector_var):
        """Test forward() returns the actual parameter."""
        vector_var.value.data = torch.ones(5) * 3
        result = vector_var.forward()
        assert torch.equal(result, torch.ones(5) * 3)
        assert result is vector_var.value

    # ----------------------
    # String representation tests
    # ----------------------

    def test_string_representations(self, vector_var):
        """Test __repr__ and __str__ contain key information."""
        repr_str = repr(vector_var)
        str_str = str(vector_var)

        assert "vector" in repr_str and "shape=(5,)" in repr_str
        assert "vector" in str_str and "torch.Size([5])" in str_str

    # ----------------------
    # State dict tests
    # ----------------------

    def test_state_dict_uses_custom_names(self, vector_var):
        """Test state_dict uses variable's custom name for parameters."""
        state = vector_var.state_dict()
        assert "vector" in state

    def test_multiple_variables_in_expression_have_distinct_names(self):
        """Test multiple variables maintain distinct names in state_dict."""
        from rlaopt.expression import expr_types

        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = expr_types.sum_expr()(x, y)

        state = expr.state_dict()
        assert any("x" in key for key in state.keys())
        assert any("y" in key for key in state.keys())
