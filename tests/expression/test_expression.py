"""Tests for Expression class."""

import pytest
import torch

from rlaopt.expression import expr_types
from rlaopt.expression.expression import Expression
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def concrete_expression():
    """Create a minimal concrete Expression for testing."""

    class ConcreteExpression(Expression):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        def is_smooth(self):
            return True

        def is_proxable(self):
            return False

        def forward(self):
            return self.value

    return ConcreteExpression()


class TestExpression:
    """Test concrete methods implemented in Expression ABC."""

    # ----------------------
    # Parameter management tests
    # ----------------------

    def test_params_names(self, concrete_expression):
        """Test params_names returns correct names."""
        names = concrete_expression.get_params_names()

        assert names[0] == "value"

    def test_params_shapes(self, concrete_expression):
        """Test params_shapes returns correct shapes."""
        shapes = concrete_expression.get_params_shapes()

        assert shapes[0] == (3,)

    def test_params_from_tensors_with_correct_shapes(self, concrete_expression):
        """Test params_from_tensors with correct shapes."""
        new_params_tensor = torch.zeros(
            3,
        )
        new_params = TensorDict({"value": new_params_tensor})

        output_ = concrete_expression.params_from_tensors((new_params_tensor,))

        assert torch.allclose(output_["value"], new_params["value"])

    def test_params_from_tensors_with_wrong_shapes(self, concrete_expression):
        """Test params_from_tensors with wrong shapes."""
        new_params_tensor = torch.ones(20, 5)
        with pytest.raises(ValueError):
            concrete_expression.params_from_tensors(new_params_tensor)

    def test_params_from_tensors_with_wrong_lens(self, concrete_expression):
        """Test params_from_tensors with wrong lengths."""
        new_params_tensors = (
            torch.zeros(
                3,
            ),
            torch.ones(20, 5),
        )
        with pytest.raises(ValueError):
            concrete_expression.params_from_tensors(new_params_tensors)

    def test_evaluate_with_different_params(self, concrete_expression):
        """Test evaluate() uses provided params without modifying stored ones."""
        original_value = concrete_expression.value.data.clone()
        param_name = list(concrete_expression.named_parameters())[0][0]
        new_params = TensorDict({param_name: torch.tensor([5.0, 6.0, 7.0])})

        result = concrete_expression.evaluate(new_params)

        assert torch.equal(result, torch.tensor([5.0, 6.0, 7.0]))
        assert torch.equal(concrete_expression.value.data, original_value)

    def test_params_property_returns_params_dict(self, concrete_expression):
        """Test params property is alias for TensorDict(params_dict())."""
        assert concrete_expression.params.to_dict() == dict(
            concrete_expression.named_parameters()
        )

    def test_update_params_modifies_stored_values(self, concrete_expression):
        """Test update_params() changes stored parameter values."""
        param_name = list(concrete_expression.named_parameters())[0][0]
        new_values = TensorDict({param_name: torch.tensor([10.0, 11.0, 12.0])})

        concrete_expression.update_params(new_values)

        assert torch.equal(concrete_expression.value, torch.tensor([10.0, 11.0, 12.0]))

    # ----------------------
    # Operator overload tests - verify correct types returned
    # ----------------------

    def test_add_returns_add_expression(self, concrete_expression):
        """Test __add__ creates an AddExpression."""
        result = concrete_expression + 5
        assert isinstance(result, expr_types.add_expr())

    def test_radd_returns_add_expression(self, concrete_expression):
        """Test __radd__ creates an AddExpression."""
        result = 5 + concrete_expression
        assert isinstance(result, expr_types.add_expr())

    def test_sub_returns_add_expression(self, concrete_expression):
        """Test __sub__ creates an AddExpression (__sub__ is __add__ with negation)."""
        result = concrete_expression - 3
        assert isinstance(result, expr_types.add_expr())

    def test_rsub_returns_add_expression(self, concrete_expression):
        """Test __rsub__ creates an AddExpression."""
        result = 10 - concrete_expression
        assert isinstance(result, expr_types.add_expr())

    def test_neg_returns_product_expression(self, concrete_expression):
        """Test __neg__ creates a ProductExpression (multiplication by -1)."""
        result = -concrete_expression
        assert isinstance(result, expr_types.prod_expr())

    def test_mul_returns_product_expression(self, concrete_expression):
        """Test __mul__ creates a ProductExpression."""
        result = concrete_expression * 2
        assert isinstance(result, expr_types.prod_expr())

    def test_rmul_returns_product_expression(self, concrete_expression):
        """Test __rmul__ creates a ProductExpression."""
        result = 2 * concrete_expression
        assert isinstance(result, expr_types.prod_expr())

    def test_truediv_with_scalar_returns_product_expression(self, concrete_expression):
        """Test __truediv__ with scalar creates a ProductExpression."""
        result = concrete_expression / 2.0
        assert isinstance(result, expr_types.prod_expr())

    def test_truediv_with_non_scalar_returns_not_implemented(self, concrete_expression):
        """Test __truediv__ with non-scalar raises TypeError."""
        with pytest.raises(TypeError):
            concrete_expression / concrete_expression

    def test_matmul_returns_product_expression(self, concrete_expression):
        """Test __matmul__ creates a ProductExpression with matmul=True."""
        result = concrete_expression @ torch.ones(3, 2)
        assert isinstance(result, expr_types.prod_expr())

    def test_rmatmul_returns_product_expression(self, concrete_expression):
        """Test __rmatmul__ creates a ProductExpression with matmul=True."""
        result = torch.ones(2, 3) @ concrete_expression
        assert isinstance(result, expr_types.prod_expr())

    def test_pow_returns_unary_op_expression(self, concrete_expression):
        """Test __pow__ creates a UnaryOpExpression."""
        result = concrete_expression**2
        assert isinstance(result, expr_types.unary_op_expr())
