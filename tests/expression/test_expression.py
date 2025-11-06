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
            self.x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
            self.y = expr_types.variable()(torch.tensor([4.0, 5.0]), name="y")

        def is_smooth(self):
            return True

        def is_proxable(self):
            return False

        def forward(self):
            return torch.sum(self.x.value**2) + torch.sum(self.y.value**2)

    return ConcreteExpression()


class TestExpression:
    """Test concrete methods implemented in Expression ABC."""

    # ----------------------
    # Variable management tests
    # ----------------------

    @pytest.mark.parametrize(
        "new_variables,expected_result",
        [
            (
                TensorDict({"x": torch.tensor([5.0, 6.0, 7.0])}),
                torch.tensor(151.0),
            ),
            (
                TensorDict(
                    {"x": torch.tensor([0.0, 0.0, 0.0]), "y": torch.tensor([8.0, 9.0])}
                ),
                torch.tensor(145.0),
            ),
        ],
        ids=["partial_update", "full_update"],
    )
    def test_evaluate_with_different_variables(
        self, concrete_expression, new_variables, expected_result
    ):
        """Test evaluate() uses provided variables without modifying stored ones."""
        original_values = {
            k: v.clone() for k, v in concrete_expression.variables_dict.items()
        }

        result = concrete_expression.evaluate(new_variables)

        assert torch.equal(result, expected_result)
        assert torch.equal(concrete_expression.x.value.data, original_values["x"])
        assert torch.equal(concrete_expression.y.value.data, original_values["y"])

    def test_variables_dict_property(self, concrete_expression):
        """Test variables_dict property returns variable values."""
        var_dict = concrete_expression.variables_dict

        assert list(var_dict.keys()) == ["x", "y"]
        assert torch.equal(var_dict["x"], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(var_dict["y"], torch.tensor([4.0, 5.0]))

    def test_variable_names(self, concrete_expression):
        """Test get_variable_names returns correct names."""
        names = concrete_expression.get_variable_names()

        assert names == ["x", "y"]

    def test_variable_shapes(self, concrete_expression):
        """Test get_variable_shapes returns correct shapes."""
        shapes = concrete_expression.get_variable_shapes()

        assert shapes == {"x": torch.Size([3]), "y": torch.Size([2])}

    @pytest.mark.parametrize(
        "input_dict,expected_keys",
        [
            (
                TensorDict(
                    {
                        "x": torch.tensor([1.0, 2.0, 3.0]),
                        "y": torch.tensor([4.0, 5.0]),
                    }
                ),
                ["x", "y"],
            ),
            (
                TensorDict(
                    {
                        "x": torch.tensor([1.0, 2.0, 3.0]),
                        "y": torch.tensor([4.0, 5.0]),
                        "z": torch.tensor([6.0, 7.0, 8.0]),
                    }
                ),
                ["x", "y"],
            ),
            (
                TensorDict({"x": torch.tensor([1.0, 2.0, 3.0])}),
                ["x"],
            ),
            (
                TensorDict({"z": torch.tensor([1.0, 2.0, 3.0])}),
                [],
            ),
        ],
        ids=[
            "exact_match",
            "extra_variables",
            "partial_match",
            "no_match",
        ],
    )
    def test_select_relevant_variables(
        self, concrete_expression, input_dict, expected_keys
    ):
        """Test select_relevant_variables filters to expression's variables."""
        result = concrete_expression.select_relevant_variables(input_dict)

        assert list(result.keys()) == expected_keys
        for key in expected_keys:
            assert torch.equal(result[key], input_dict[key])

    @pytest.mark.parametrize(
        "new_values,expected_x,expected_y",
        [
            (
                TensorDict({"x": torch.tensor([10.0, 11.0, 12.0])}),
                torch.tensor([10.0, 11.0, 12.0]),
                torch.tensor([4.0, 5.0]),  # y should remain unchanged
            ),
            (
                TensorDict(
                    {
                        "x": torch.tensor([10.0, 11.0, 12.0]),
                        "y": torch.tensor([13.0, 14.0]),
                    }
                ),
                torch.tensor([10.0, 11.0, 12.0]),
                torch.tensor([13.0, 14.0]),
            ),
        ],
        ids=["partial_update", "full_update"],
    )
    def test_update_variables_modifies_stored_values(
        self, concrete_expression, new_values, expected_x, expected_y
    ):
        """Test update_variables() changes stored variable values."""
        concrete_expression.update_variables(new_values)

        assert torch.equal(concrete_expression.x.value, expected_x)
        assert torch.equal(concrete_expression.y.value, expected_y)

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
