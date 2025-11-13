"""Tests for Expression class."""

import pytest
import torch

from rlaopt.atoms import L1Norm, SumSquares
from rlaopt.expression import ExprTree, expr_types
from rlaopt.expression.expression import Expression
from rlaopt.ext_tensordict import TensorDict


def _assert_expression_equals(result, expected_tree, expected_value):
    """Helper to assert expression has correct tree and forward value."""
    assert result.tree() == expected_tree
    assert torch.allclose(result.forward(), expected_value)


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

        def tree(self):
            raise NotImplementedError()

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
            (
                TensorDict(
                    {
                        "x": torch.tensor([2.0, 3.0, 4.0]),
                        "y": torch.tensor([1.0, 1.0]),
                        "z": torch.tensor([100.0, 200.0]),  # irrelevant variable
                    }
                ),
                torch.tensor(31.0),
            ),
        ],
        ids=["partial_update", "full_update", "extra_variables"],
    )
    def test_evaluate_with_different_variables(
        self, concrete_expression, new_variables, expected_result
    ):
        """Test evaluate() uses provided variables without modifying stored ones."""
        original_values = {
            k: v.clone() for k, v in concrete_expression.variable_values.items()
        }

        result = concrete_expression.evaluate(new_variables)

        assert torch.equal(result, expected_result)
        assert torch.equal(concrete_expression.x.value.data, original_values["x"])
        assert torch.equal(concrete_expression.y.value.data, original_values["y"])

    def test_variable_values_property(self, concrete_expression):
        """Test variable_values property returns variable values."""
        var_dict = concrete_expression.variable_values

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
            (
                TensorDict(
                    {
                        "x": torch.tensor([7.0, 8.0, 9.0]),
                        "y": torch.tensor([2.0, 3.0]),
                        "z": torch.tensor([999.0]),  # irrelevant variable
                    }
                ),
                torch.tensor([7.0, 8.0, 9.0]),
                torch.tensor([2.0, 3.0]),
            ),
        ],
        ids=["partial_update", "full_update", "extra_variables"],
    )
    def test_update_variables_modifies_stored_values(
        self, concrete_expression, new_values, expected_x, expected_y
    ):
        """Test update_variables() changes stored variable values."""
        concrete_expression.update_variables(new_values)

        assert torch.equal(concrete_expression.x.value, expected_x)
        assert torch.equal(concrete_expression.y.value, expected_y)

    # ----------------------
    # Operator overload tests - verify structure and values
    # ----------------------

    def test_constant_folding_addition(self):
        """Test that constant addition folds into single constant."""
        a = expr_types.constant()(1.0)
        b = expr_types.constant()(2.0)
        c = expr_types.constant()(3.0)

        result = a + b + c

        _assert_expression_equals(
            result, ExprTree("ConstExpression"), torch.tensor(6.0)
        )

    def test_addition_flattening(self):
        """Test that (a + b) + c flattens to AddExpression(a, b, c)."""
        a = expr_types.constant()(1.0)
        b = expr_types.constant()(2.0)
        # Use variable to prevent constant folding
        x = expr_types.variable()(torch.tensor(3.0), name="x")

        result = (a + b) + x

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree("ConstExpression"),
                ExprTree("Variable(x)"),
                is_commutative=True,
            ),
            torch.tensor(6.0),
        )

    def test_zero_elimination_in_addition(self):
        """Test that expr + 0 simplifies to expr."""
        x = expr_types.variable()(torch.tensor(5.0), name="x")
        zero = expr_types.constant()(0.0)

        result = x + zero

        _assert_expression_equals(result, ExprTree("Variable(x)"), torch.tensor(5.0))

    def test_constants_cancel_to_zero(self):
        """Test that constants that sum to zero are eliminated."""
        x = expr_types.variable()(torch.tensor(5.0), name="x")
        a = expr_types.constant()(3.0)
        b = expr_types.constant()(-3.0)

        result = x + a + b

        _assert_expression_equals(result, ExprTree("Variable(x)"), torch.tensor(5.0))

    def test_broadcast_constant_matrix_plus_scalar(self):
        """Test that constant matrix + constant scalar broadcasts and folds."""
        matrix = expr_types.constant()(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        scalar = expr_types.constant()(10.0)

        result = matrix + scalar

        _assert_expression_equals(
            result,
            ExprTree("ConstExpression"),
            torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
        )

    def test_broadcast_constant_matrix_plus_vector(self):
        """Test that constant matrix + constant vector broadcasts and folds."""
        matrix = expr_types.constant()(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        vector = expr_types.constant()(torch.tensor([10.0, 20.0]))

        result = matrix + vector

        _assert_expression_equals(
            result,
            ExprTree("ConstExpression"),
            torch.tensor([[11.0, 22.0], [13.0, 24.0]]),
        )

    def test_subtraction_as_negation(self):
        """Test that a - b becomes a + (-b)."""
        a = expr_types.constant()(10.0)
        b = expr_types.constant()(3.0)

        result = a - b

        _assert_expression_equals(
            result, ExprTree("ConstExpression"), torch.tensor(7.0)
        )

    def test_negation(self):
        """Test that -expr creates ProductExpression(-1, expr)."""
        x = expr_types.variable()(torch.tensor(5.0), name="x")

        result = -x

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("ConstExpression"),
                ExprTree("Variable(x)"),
                is_commutative=True,
            ),
            torch.tensor(-5.0),
        )

    def test_multiplication_flattening(self):
        """Test that (a * 2) * 3 flattens and folds constants."""
        x = expr_types.variable()(torch.tensor(5.0), name="x")

        result = (x * 2) * 3

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("ConstExpression"),
                ExprTree("Variable(x)"),
                is_commutative=True,
            ),
            torch.tensor(30.0),
        )

    def test_distribution_over_addition_constants(self):
        """Test that 2 * (a + b) turns into a constant for constant inputs."""
        a = expr_types.constant()(3.0)
        b = expr_types.constant()(5.0)

        result = 2 * (a + b)

        _assert_expression_equals(
            result, ExprTree("ConstExpression"), torch.tensor(16.0)
        )

    def test_division_as_multiplication(self):
        """Test that expr / 2 becomes expr * 0.5."""
        x = expr_types.variable()(torch.tensor(10.0), name="x")

        result = x / 2.0

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("ConstExpression"),
                ExprTree("Variable(x)"),
                is_commutative=True,
            ),
            torch.tensor(5.0),
        )

    def test_truediv_with_non_scalar_raises_error(self):
        """Test that dividing by non-scalar raises TypeError."""
        x = expr_types.variable()(torch.tensor(10.0), name="x")
        y = expr_types.variable()(torch.tensor(2.0), name="y")

        with pytest.raises(TypeError):
            x / y

    def test_pow_creates_unary_op(self):
        """Test that expr**2 creates UnaryOpExpression."""
        x = expr_types.variable()(torch.tensor(3.0), name="x")

        result = x**2

        _assert_expression_equals(
            result,
            ExprTree("power_2", ExprTree("Variable(x)")),
            torch.tensor(9.0),
        )

    def test_distribution_over_addition_variables(self):
        """Test that 2 * (x + y) distributes to 2*x + 2*y for variables."""
        x = expr_types.variable()(torch.tensor(3.0), name="x")
        y = expr_types.variable()(torch.tensor(5.0), name="y")

        result = 2 * (x + y)

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(x)"),
                    is_commutative=True,
                ),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(y)"),
                    is_commutative=True,
                ),
                is_commutative=True,
            ),
            torch.tensor(16.0),
        )

    def test_matmul_no_optimization(self):
        """Test that matrix multiplication is not optimized."""
        a = expr_types.variable()(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), name="a")
        b = expr_types.variable()(torch.tensor([[5.0, 6.0], [7.0, 8.0]]), name="b")

        result = a @ b

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("Variable(a)"),
                ExprTree("Variable(b)"),
            ),
            torch.tensor([[19.0, 22.0], [43.0, 50.0]]),
        )

    def test_matmul_no_flattening(self):
        """Test that (a @ b) @ c does not flatten."""
        a = expr_types.variable()(torch.tensor([[1.0, 2.0]]), name="a")  # 1x2
        b = expr_types.variable()(torch.tensor([[3.0], [4.0]]), name="b")  # 2x1
        c = expr_types.variable()(torch.tensor([5.0]), name="c")

        result = (a @ b) @ c

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("Variable(a)"),
                    ExprTree("Variable(b)"),
                ),
                ExprTree("Variable(c)"),
            ),
            torch.tensor([[55.0]]),
        )

    def test_scalar_times_matmul_no_distribution(self):
        """Test that 2 * (a @ b) does not distribute for matmul."""
        a = expr_types.variable()(torch.tensor([[1.0, 2.0]]), name="a")
        b = expr_types.variable()(torch.tensor([[3.0], [4.0]]), name="b")

        result = 2 * (a @ b)

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("ConstExpression"),
                ExprTree(
                    "ProductExpression",
                    ExprTree("Variable(a)"),
                    ExprTree("Variable(b)"),
                ),
                is_commutative=True,
            ),
            torch.tensor([[22.0]]),
        )

    def test_matmul_constants_creates_product_expression(self):
        """Test that matmul between constants creates ProductExpression (not folded)."""
        a = expr_types.constant()(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        b = expr_types.constant()(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

        result = a @ b

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("ConstExpression"),
                ExprTree("ConstExpression"),
            ),
            torch.tensor([[19.0, 22.0], [43.0, 50.0]]),
        )

    def test_matmul_no_distribution_over_sum(self):
        """Test that A @ (x + y) does not distribute to (A @ x) + (A @ y)."""
        A = expr_types.constant()(torch.tensor([[1.0, 2.0]]))
        x = expr_types.variable()(torch.tensor([[3.0], [4.0]]), name="x")
        y = expr_types.variable()(torch.tensor([[5.0], [6.0]]), name="y")

        result = A @ (x + y)

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("ConstExpression"),
                ExprTree(
                    "AddExpression",
                    ExprTree("Variable(x)"),
                    ExprTree("Variable(y)"),
                    is_commutative=True,
                ),
            ),
            torch.tensor([[28.0]]),
        )

    def test_atom_scaling_optimization(self):
        """Test that scalar * atom calls returns the same type of atom."""
        x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
        atom = L1Norm(x, scaling=1.0)

        result = 2.0 * atom

        _assert_expression_equals(
            result,
            ExprTree("L1Norm", ExprTree("Variable(x)")),
            torch.tensor(12.0),
        )

    def test_atom_scaling_with_folded_constants(self):
        """Test that (2 * 3) * atom folds constants before scaling."""
        x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
        atom = L1Norm(x, scaling=1.0)

        result = 2.0 * (3.0 * atom)

        _assert_expression_equals(
            result,
            ExprTree("L1Norm", ExprTree("Variable(x)")),
            torch.tensor(36.0),
        )

    def test_atom_scaling_preserves_existing_scaling(self):
        """Test that scaling an already-scaled atom multiplies the scalings."""
        x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
        atom = L1Norm(x, scaling=5.0)

        result = 2.0 * atom

        _assert_expression_equals(
            result,
            ExprTree("L1Norm", ExprTree("Variable(x)")),
            torch.tensor(60.0),
        )

    def test_atom_without_scale_falls_back_to_product(self):
        """Test that atoms without _scale() support fall back to ProductExpression."""
        x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
        atom = SumSquares(x)

        result = 2.0 * atom

        _assert_expression_equals(
            result,
            ExprTree(
                "ProductExpression",
                ExprTree("ConstExpression"),
                ExprTree("SumSquares", ExprTree("Variable(x)")),
                is_commutative=True,
            ),
            torch.tensor(28.0),
        )

    # ----------------------
    # Stress tests - complex combinations
    # ----------------------

    def test_mixed_scalar_mult_and_addition(self):
        """Test 2*x + 3*y + 4*z with multiple scalar multiplications and additions."""
        x = expr_types.variable()(torch.tensor(1.0), name="x")
        y = expr_types.variable()(torch.tensor(2.0), name="y")
        z = expr_types.variable()(torch.tensor(3.0), name="z")

        result = 2 * x + 3 * y + 4 * z

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(x)"),
                    is_commutative=True,
                ),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(y)"),
                    is_commutative=True,
                ),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(z)"),
                    is_commutative=True,
                ),
                is_commutative=True,
            ),
            torch.tensor(20.0),
        )

    def test_nested_distribution(self):
        """Test 2 * (3 * x + 4 * y) with nested distribution."""
        x = expr_types.variable()(torch.tensor(1.0), name="x")
        y = expr_types.variable()(torch.tensor(2.0), name="y")

        result = 2 * (3 * x + 4 * y)

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(x)"),
                    is_commutative=True,
                ),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(y)"),
                    is_commutative=True,
                ),
                is_commutative=True,
            ),
            torch.tensor(22.0),
        )

    def test_matmul_with_scalar_mult_and_addition(self):
        """Test 2*(A@x) + 3*(B@y) with matmul and scalar multiplication."""
        A = expr_types.constant()(torch.tensor([[1.0, 2.0]]))
        B = expr_types.constant()(torch.tensor([[3.0, 4.0]]))
        x = expr_types.variable()(torch.tensor([[5.0], [6.0]]), name="x")
        y = expr_types.variable()(torch.tensor([[7.0], [8.0]]), name="y")

        result = 2 * (A @ x) + 3 * (B @ y)

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree(
                        "ProductExpression",
                        ExprTree("ConstExpression"),
                        ExprTree("Variable(x)"),
                    ),
                    is_commutative=True,
                ),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree(
                        "ProductExpression",
                        ExprTree("ConstExpression"),
                        ExprTree("Variable(y)"),
                    ),
                    is_commutative=True,
                ),
                is_commutative=True,
            ),
            torch.tensor([[193.0]]),
        )

    def test_complex_mix_matmul_scalar_add(self):
        """Test 2*(A@x) + 3*y + 4 mixing matmul, scalar mult, and constants."""
        A = expr_types.constant()(torch.tensor([[1.0, 2.0]]))
        x = expr_types.variable()(torch.tensor([[3.0], [4.0]]), name="x")
        y = expr_types.variable()(torch.tensor([[5.0]]), name="y")

        result = 2 * (A @ x) + 3 * y + 4

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree(
                        "ProductExpression",
                        ExprTree("ConstExpression"),
                        ExprTree("Variable(x)"),
                    ),
                    is_commutative=True,
                ),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(y)"),
                    is_commutative=True,
                ),
                ExprTree("ConstExpression"),
                is_commutative=True,
            ),
            torch.tensor([[41.0]]),
        )

    def test_atom_with_scalar_mult_and_addition(self):
        """Test 2*L1Norm(x) + 3*y mixing atom scaling with other operations."""
        x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = expr_types.variable()(torch.tensor(5.0), name="y")
        atom = L1Norm(x, scaling=1.0)

        result = 2 * atom + 3 * y

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree("L1Norm", ExprTree("Variable(x)")),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(y)"),
                    is_commutative=True,
                ),
                is_commutative=True,
            ),
            torch.tensor(27.0),
        )

    def test_distribution_with_constants_and_matmul(self):
        """Test 2*(x + 3) + (A@y) mixing distribution, constants, and matmul."""
        x = expr_types.variable()(torch.tensor(5.0), name="x")
        A = expr_types.constant()(torch.tensor([[1.0, 2.0]]))
        y = expr_types.variable()(torch.tensor([[3.0], [4.0]]), name="y")

        result = 2 * (x + 3) + (A @ y)

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(x)"),
                    is_commutative=True,
                ),
                ExprTree("ConstExpression"),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(y)"),
                ),
                is_commutative=True,
            ),
            torch.tensor([[27.0]]),
        )

    def test_deeply_nested_operations(self):
        """Test ((2*x + 3) * 4 + 5*y) * 6 with deep nesting."""
        x = expr_types.variable()(torch.tensor(1.0), name="x")
        y = expr_types.variable()(torch.tensor(2.0), name="y")

        result = ((2 * x + 3) * 4 + 5 * y) * 6

        _assert_expression_equals(
            result,
            ExprTree(
                "AddExpression",
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(x)"),
                    is_commutative=True,
                ),
                ExprTree(
                    "ProductExpression",
                    ExprTree("ConstExpression"),
                    ExprTree("Variable(y)"),
                    is_commutative=True,
                ),
                ExprTree("ConstExpression"),
                is_commutative=True,
            ),
            torch.tensor(180.0),
        )
