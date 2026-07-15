"""Tests for QuadForm atom."""

import pytest
import torch

from rlaopt.atoms.quad_form import QuadForm
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    x = Variable(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
    return x


@pytest.fixture
def diag_Q():
    """Diagonal matrix diag(1, 2, 3, 4, 5)."""
    return torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


@pytest.fixture
def simple_affine(vector_var):
    """Create a simple (identity) affine transformation."""
    b = torch.zeros(5)
    return vector_var + b


@pytest.fixture
def nontrivial_affine(vector_var):
    """Create a non-trivial affine transformation producing a 3-vector."""
    A = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    b = torch.tensor([1.0, -1.0, 0.5])
    return A @ vector_var + b


class TestQuadFormInitialization:
    """Test QuadForm initialization and Q validation."""

    def test_init_with_variable(self, vector_var, diag_Q):
        """Test initialization with Variable input."""
        qf = QuadForm(vector_var, diag_Q)
        assert isinstance(qf.get_input("x"), Variable)

    def test_init_with_expression(self, simple_affine, diag_Q):
        """Test initialization with Expression input."""
        qf = QuadForm(simple_affine, diag_Q)
        assert isinstance(qf.get_input("x"), Expression)

    def test_q_stored_as_buffer(self, vector_var, diag_Q):
        """Test that Q is stored as a tensor buffer."""
        qf = QuadForm(vector_var, diag_Q)
        Q = qf.get_buffer("Q")
        assert isinstance(Q, torch.Tensor)
        assert torch.equal(Q, diag_Q)

    @pytest.mark.parametrize(
        "bad_Q, match",
        [
            (torch.ones(5), "2-dimensional"),
            (torch.tensor(1.0), "2-dimensional"),
            (torch.ones(2, 2, 2), "2-dimensional"),
            (torch.ones(2, 3), "square"),
        ],
        ids=["1d", "scalar", "3d", "non-square"],
    )
    def test_invalid_q_raises(self, vector_var, bad_Q, match):
        """Test that a non-2D or non-square Q raises ValueError."""
        with pytest.raises(ValueError, match=match):
            QuadForm(vector_var, bad_Q)


class TestQuadFormForward:
    """Test QuadForm forward evaluation."""

    def test_forward_general_q(self):
        """Test forward() computes x^T Q x for a general matrix."""
        x = Variable(torch.tensor([1.0, 2.0]), name="x")
        Q = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        # Q @ [1,2] = [4, 7]; [1,2] . [4,7] = 18
        result = QuadForm(x, Q).forward()
        assert torch.allclose(result, torch.tensor(18.0))

    def test_forward_diagonal_q(self, vector_var, diag_Q):
        """Test forward() with a diagonal Q is a weighted sum of squares."""
        result = QuadForm(vector_var, diag_Q).forward()
        # sum(i * x_i^2) over i,x_i in [1..5] = 1+8+27+64+125 = 225
        assert torch.allclose(result, torch.tensor(225.0))

    def test_forward_with_affine_expression(self, nontrivial_affine):
        """Test forward() with an affine-expression input."""
        # A @ x + b = [2, 3, 3.5]; with identity Q -> 4 + 9 + 12.25 = 25.25
        result = QuadForm(nontrivial_affine, torch.eye(3)).forward()
        assert torch.allclose(result, torch.tensor(25.25))

    def test_forward_with_zero_variable(self, diag_Q):
        """Test forward() with a zero variable is zero."""
        x = Variable(torch.zeros(5), name="x")
        result = QuadForm(x, diag_Q).forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_asymmetric_q_uses_q_as_given(self):
        """Test forward() uses Q as given (equal to its symmetric part)."""
        x = Variable(torch.tensor([1.0, 1.0]), name="x")
        Q_asym = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        # x^T Q x = 1 + 2 + 0 + 1 = 4
        result = QuadForm(x, Q_asym).forward()
        assert torch.allclose(result, torch.tensor(4.0))
        # x^T Q x == x^T ((Q + Q^T)/2) x
        Q_sym = (Q_asym + Q_asym.T) / 2
        assert torch.allclose(result, QuadForm(x, Q_sym).forward())


class TestQuadFormProperties:
    """Test QuadForm smoothness and proxability."""

    def test_is_smooth_with_variable(self, vector_var, diag_Q):
        """Test is_smooth() returns True for Variable input."""
        assert QuadForm(vector_var, diag_Q).is_smooth() is True

    def test_is_smooth_with_smooth_expression(self, simple_affine, diag_Q):
        """Test is_smooth() returns True for a smooth expression."""
        assert QuadForm(simple_affine, diag_Q).is_smooth() is True

    def test_is_smooth_with_nonsmooth_expression(self):
        """Test is_smooth() returns False for a non-smooth expression."""

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

        qf = QuadForm(NonSmoothExpr(), torch.eye(5))
        assert qf.is_smooth() is False

    def test_is_proxable_returns_false(self, vector_var, diag_Q):
        """Test is_proxable() returns False even with a Variable input."""
        assert QuadForm(vector_var, diag_Q).is_proxable() is False


class TestQuadFormProx:
    """Test that QuadForm prox is not implemented."""

    def test_prox_raises_not_implemented(self, vector_var, diag_Q):
        """Test prox() raises NotImplementedError."""
        qf = QuadForm(vector_var, diag_Q)
        with pytest.raises(NotImplementedError, match="Proximal operator"):
            qf.prox(TensorDict({vector_var.name: torch.ones(5)}), 1.0)


class TestQuadFormDecompose:
    """Test QuadForm decompose method."""

    def test_decompose_with_variable_returns_none(self, vector_var, diag_Q):
        """Test decompose returns None for a Variable input."""
        assert QuadForm(vector_var, diag_Q).decompose() is None

    def test_decompose_with_affine_returns_none(self, simple_affine, diag_Q):
        """Test decompose returns None for an affine input."""
        assert QuadForm(simple_affine, diag_Q).decompose() is None
