"""Tests for LinSys module."""

import pytest
import torch

from rlaopt.scientific_computing.lin_sys import LinSys


@pytest.fixture
def valid_A():
    """Fixture for a valid positive-definite matrix A."""
    return torch.tensor([[2.0, 0.5], [0.5, 3.0]])


@pytest.fixture
def valid_B_1d():
    """Fixture for a valid 1D right-hand side vector B."""
    return torch.tensor([1.0, 2.0])


@pytest.fixture
def valid_B_2d():
    """Fixture for a valid 2D right-hand side matrix B."""
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def valid_reg():
    """Fixture for a valid regularization parameter."""
    return 0.5


class TestLinSysInitialization:
    """Test successful initialization of LinSys."""

    def test_init_with_1d_B(self, valid_A, valid_B_1d):
        """Test initialization with 1D B vector."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d)
        assert lin_sys is not None

    def test_init_with_2d_B(self, valid_A, valid_B_2d):
        """Test initialization with 2D B matrix."""
        lin_sys = LinSys(A=valid_A, B=valid_B_2d)
        assert lin_sys is not None

    def test_init_with_regularization(self, valid_A, valid_B_1d, valid_reg):
        """Test initialization with regularization parameter."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=valid_reg)
        assert lin_sys is not None

    def test_init_with_zero_regularization(self, valid_A, valid_B_1d):
        """Test initialization with zero regularization (default)."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=0.0)
        assert lin_sys is not None


class TestLinSysProperties:
    """Test properties of LinSys."""

    def test_property_A(self, valid_A, valid_B_1d):
        """Test that A property returns the correct matrix."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d)
        assert torch.equal(lin_sys.A, valid_A)

    def test_property_B(self, valid_A, valid_B_1d):
        """Test that B property returns the correct vector."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d)
        assert torch.equal(lin_sys.B, valid_B_1d)

    def test_property_reg(self, valid_A, valid_B_1d, valid_reg):
        """Test that reg property returns the correct value."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=valid_reg)
        assert lin_sys.reg == valid_reg

    def test_property_reg_default(self, valid_A, valid_B_1d):
        """Test that reg property returns default value when not specified."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d)
        assert lin_sys.reg == 0.0


class TestLinSysCall:
    """Test __call__ method of LinSys."""

    def test_call_without_regularization(self, valid_A, valid_B_1d):
        """Test calling LinSys without regularization."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=0.0)
        v = torch.tensor([1.0, 1.0])
        result = lin_sys(v)
        expected = valid_A @ v
        assert torch.allclose(result, expected)

    def test_call_with_regularization(self, valid_A, valid_B_1d):
        """Test calling LinSys with regularization."""
        reg = 0.5
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=reg)
        v = torch.tensor([1.0, 1.0])
        result = lin_sys(v)
        expected = valid_A @ v + reg * v
        assert torch.allclose(result, expected)


class TestLinSysCheckInputsErrors:
    """Test all error conditions in _check_inputs."""

    def test_A_not_tensor(self, valid_B_1d):
        """Test error when A is not a torch.Tensor."""
        A_invalid = [[1.0, 0.0], [0.0, 1.0]]  # List instead of tensor
        with pytest.raises(TypeError, match="A must be a torch.Tensor"):
            LinSys(A=A_invalid, B=valid_B_1d)

    def test_B_not_tensor(self, valid_A):
        """Test error when B is not a torch.Tensor."""
        B_invalid = [1.0, 2.0]  # List instead of tensor
        with pytest.raises(TypeError, match="B must be a torch.Tensor"):
            LinSys(A=valid_A, B=B_invalid)

    def test_A_not_square(self, valid_B_1d):
        """Test error when A is not a square matrix."""
        A_invalid = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="A must be a square matrix"):
            LinSys(A=A_invalid, B=valid_B_1d)

    def test_A_not_2d(self, valid_B_1d):
        """Test error when A is not 2D."""
        A_invalid = torch.tensor([1.0, 2.0, 3.0])  # 1D tensor
        with pytest.raises(ValueError, match="A must be a square matrix"):
            LinSys(A=A_invalid, B=valid_B_1d)

    def test_B_wrong_dimension(self, valid_A):
        """Test error when B has wrong number of dimensions."""
        B_invalid = torch.tensor([[[1.0, 2.0]]])  # 3D tensor
        with pytest.raises(
            ValueError,
            match="B must be a tensor whose first dimension matches A's size",
        ):
            LinSys(A=valid_A, B=B_invalid)

    def test_B_size_mismatch_1d(self, valid_A):
        """Test error when 1D B size doesn't match A."""
        B_invalid = torch.tensor([1.0, 2.0, 3.0])  # Size 3, A is 2x2
        with pytest.raises(
            ValueError,
            match="B must be a tensor whose first dimension matches A's size",
        ):
            LinSys(A=valid_A, B=B_invalid)

    def test_B_size_mismatch_2d(self, valid_A):
        """Test error when 2D B first dimension doesn't match A."""
        B_invalid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2, A is 2x2
        with pytest.raises(
            ValueError,
            match="B must be a tensor whose first dimension matches A's size",
        ):
            LinSys(A=valid_A, B=B_invalid)

    def test_reg_not_float(self, valid_A, valid_B_1d):
        """Test error when reg is not a float."""
        with pytest.raises(ValueError, match="reg must be a non-negative float"):
            LinSys(A=valid_A, B=valid_B_1d, reg=1)  # int instead of float

    def test_reg_negative(self, valid_A, valid_B_1d):
        """Test error when reg is negative."""
        with pytest.raises(ValueError, match="reg must be a non-negative float"):
            LinSys(A=valid_A, B=valid_B_1d, reg=-0.5)

    def test_reg_not_numeric(self, valid_A, valid_B_1d):
        """Test error when reg is not numeric."""
        with pytest.raises(ValueError, match="reg must be a non-negative float"):
            LinSys(A=valid_A, B=valid_B_1d, reg="0.5")  # string instead of float
