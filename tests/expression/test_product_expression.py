import pytest
import torch

from rlaopt.expression import (
    AddExpression,
    ConstExpression,
    Expression,
    ExprTree,
    Variable,
)
from rlaopt.expression.op_expressions import ProductExpression


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    x = Variable((5,), name="x")
    x.value.data = torch.ones(5) * 2
    return x


@pytest.fixture
def another_vector_var():
    """Create another vector variable."""
    y = Variable((5,), name="y")
    y.value.data = torch.ones(5) * 3
    return y


@pytest.fixture
def matrix_var():
    """Create a matrix variable."""
    A = Variable((3, 4), name="A")
    A.value.data = torch.ones(3, 4) * 2
    return A


@pytest.fixture
def vector_for_matmul():
    """Create a vector for matrix multiplication."""
    b = Variable((4,), name="b")
    b.value.data = torch.ones(4) * 3
    return b


class TestProductExpression:
    """Test ProductExpression concrete implementation."""

    # ----------------------
    # Initialization and basic operations
    # ----------------------

    def test_init_with_two_expressions(self, vector_var, another_vector_var):
        """Test initialization with two expressions."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)
        assert prod.n_exprs == 2
        assert prod.matmul is False

    def test_elementwise_multiplication(self, vector_var, another_vector_var):
        """Test elementwise multiplication operator."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)
        result = prod.op([torch.ones(5) * 2, torch.ones(5) * 3])
        assert torch.equal(result, torch.ones(5) * 6)

    def test_matrix_multiplication(self, matrix_var, vector_for_matmul):
        """Test matrix multiplication operator."""
        prod = ProductExpression(matrix_var, vector_for_matmul, matmul=True)
        A_val = torch.ones(3, 4) * 2
        b_val = torch.ones(4) * 3
        result = prod.op([A_val, b_val])
        expected = torch.matmul(A_val, b_val)
        assert torch.equal(result, expected)

    def test_op_with_single_expression(self, vector_var):
        """Test op() with single expression returns that value."""
        prod = ProductExpression(vector_var, matmul=False)
        result = prod.op([torch.ones(5) * 5])
        assert torch.equal(result, torch.ones(5) * 5)

    def test_op_with_multiple_expressions(self):
        """Test op() with more than two expressions."""
        x = Variable((3,), name="x")
        y = Variable((3,), name="y")
        z = Variable((3,), name="z")
        prod = ProductExpression(x, y, z, matmul=False)

        values = [
            torch.tensor([2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0]),
            torch.tensor([8.0, 9.0, 10.0]),
        ]
        result = prod.op(values)
        assert torch.equal(result, torch.tensor([80.0, 162.0, 280.0]))

    # ----------------------
    # Validation tests
    # ----------------------

    def test_multiply_variable_and_constant_allowed(self, vector_var):
        """Test multiplying Variable and Constant is allowed."""
        const = ConstExpression(5.0)
        prod = ProductExpression(vector_var, const, matmul=False)
        assert prod.n_exprs == 2

    def test_multiply_two_variables_allowed(self, vector_var, another_vector_var):
        """Test multiplying two Variables is allowed."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)
        assert prod.n_exprs == 2

    def test_multiply_expression_trees_of_vars_and_consts_allowed(
        self, vector_var, another_vector_var
    ):
        """Test multiplying expressions built from Variables and Constants is allowed."""
        sum_expr = AddExpression(vector_var, another_vector_var)
        const = ConstExpression(2.0)
        prod = ProductExpression(sum_expr, const, matmul=False)
        assert prod.n_exprs == 2

    def test_multiply_complex_parameterized_expressions_raises_error(self):
        """Test multiplying complex parameterized expressions raises TypeError."""
        x = Variable((5,), name="x")

        # Create a complex expression that's not just Variables/Constants/basic ops
        class ComplexExpression(Expression):
            def __init__(self, var):
                super().__init__()
                self.var = var

            def is_smooth(self):
                return True

            def is_proxable(self):
                return False

            def forward(self):
                return self.var.forward() ** 2

            def tree(self):
                raise NotImplementedError()

        complex1 = ComplexExpression(x)
        complex2 = ComplexExpression(x)

        with pytest.raises(
            TypeError, match="Cannot multiply two arbitrary parameterized"
        ):
            ProductExpression(complex1, complex2, matmul=False)

    # ----------------------
    # Smoothness and proxability tests
    # ----------------------

    def test_is_smooth_always_true(self, vector_var, another_vector_var):
        """Test products are always smooth."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)
        assert prod.is_smooth() is True

    def test_is_proxable_always_false(self, vector_var, another_vector_var):
        """Test products are not proxable."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)
        assert prod.is_proxable() is False

    def test_prox_raises_not_implemented(self, vector_var, another_vector_var):
        """Test prox() raises NotImplementedError."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)
        with pytest.raises(NotImplementedError, match="not proxable"):
            prod.prox(torch.ones(5), 1.0)

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_elementwise_multiplication(self, vector_var, another_vector_var):
        """Test forward() with elementwise multiplication."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)
        result = prod.forward()
        assert torch.equal(result, torch.ones(5) * 6)  # 2 * 3 = 6

    def test_forward_matrix_multiplication(self, matrix_var, vector_for_matmul):
        """Test forward() with matrix multiplication."""
        prod = ProductExpression(matrix_var, vector_for_matmul, matmul=True)
        result = prod.forward()
        expected = (
            torch.ones(3) * 24
        )  # (3x4 matrix of 2s) @ (4-vector of 3s) = 3-vector of 24s
        assert torch.equal(result, expected)

    # ----------------------
    # Matmul flag tests
    # ----------------------

    def test_matmul_flag_stored_correctly(self, vector_var):
        """Test matmul flag is stored correctly."""
        elem_prod = ProductExpression(vector_var, matmul=False)
        mat_prod = ProductExpression(vector_var, matmul=True)
        assert elem_prod.matmul is False
        assert mat_prod.matmul is True

    # ----------------------
    # Tree representation tests
    # ----------------------

    def test_tree_structure_elementwise(self, vector_var, another_vector_var):
        """Test tree() for elementwise multiplication (matmul=False) is commutative."""
        prod = ProductExpression(vector_var, another_vector_var, matmul=False)

        expected = ExprTree(
            "ProductExpression",
            vector_var.tree(),
            another_vector_var.tree(),
            is_commutative=True,
        )
        assert prod.tree() == expected

    def test_tree_structure_matmul(self, matrix_var, vector_for_matmul):
        """Test tree() for matrix multiplication (matmul=True) is NOT commutative."""
        prod = ProductExpression(matrix_var, vector_for_matmul, matmul=True)

        expected = ExprTree(
            "ProductExpression",
            matrix_var.tree(),
            vector_for_matmul.tree(),
            is_commutative=False,
        )
        assert prod.tree() == expected
