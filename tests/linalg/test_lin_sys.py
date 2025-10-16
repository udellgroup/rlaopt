"""Tests for LinSys module."""

import pytest
import torch

from rlaopt.linalg.lin_sys import LinSys


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


@pytest.fixture
def valid_w_1d():
    """Fixture for a valid 1D initial guess vector w."""
    return torch.tensor([0.5, 1.5])


@pytest.fixture
def valid_w_2d():
    """Fixture for a valid 2D initial guess matrix w."""
    return torch.tensor([[0.5, 1.5], [2.5, 3.5]])


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

    def test_init_with_w_1d(self, valid_A, valid_B_1d, valid_w_1d):
        """Test initialization with 1D initial guess w."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, w=valid_w_1d)
        assert lin_sys is not None
        assert torch.equal(lin_sys.w, valid_w_1d)

    def test_init_with_w_2d(self, valid_A, valid_B_2d, valid_w_2d):
        """Test initialization with 2D initial guess w."""
        lin_sys = LinSys(A=valid_A, B=valid_B_2d, w=valid_w_2d)
        assert lin_sys is not None
        assert torch.equal(lin_sys.w, valid_w_2d)

    def test_init_without_w_defaults_to_zeros(self, valid_A, valid_B_1d):
        """Test that w defaults to zeros when not provided."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d)
        expected_w = torch.zeros_like(valid_B_1d)
        assert torch.equal(lin_sys.w, expected_w)

    def test_init_without_w_2d_defaults_to_zeros(self, valid_A, valid_B_2d):
        """Test that w defaults to zeros for 2D B when not provided."""
        lin_sys = LinSys(A=valid_A, B=valid_B_2d)
        expected_w = torch.zeros_like(valid_B_2d)
        assert torch.equal(lin_sys.w, expected_w)


class TestLinSysForward:
    """Test forward method of LinSys."""

    def test_forward_without_regularization(self, valid_A, valid_B_1d):
        """Test calling LinSys without regularization."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=0.0)
        v = torch.tensor([1.0, 1.0])
        result = lin_sys(v)
        expected = valid_A @ v
        assert torch.allclose(result, expected)

    def test_forward_with_regularization(self, valid_A, valid_B_1d):
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

    def test_w_not_tensor(self, valid_A, valid_B_1d):
        """Test error when w is not a torch.Tensor."""
        w_invalid = [0.5, 1.5]
        with pytest.raises(TypeError, match="w must be a torch.Tensor"):
            LinSys(A=valid_A, B=valid_B_1d, w=w_invalid)

    def test_w_shape_mismatch_1d(self, valid_A, valid_B_1d):
        """Test error when w shape doesn't match B for 1D case."""
        w_invalid = torch.tensor([0.5, 1.5, 2.5])
        with pytest.raises(ValueError, match="w must have the same shape as B"):
            LinSys(A=valid_A, B=valid_B_1d, w=w_invalid)

    def test_w_shape_mismatch_2d(self, valid_A, valid_B_2d):
        """Test error when w shape doesn't match B for 2D case."""
        w_invalid = torch.tensor([[0.5, 1.5]])
        with pytest.raises(ValueError, match="w must have the same shape as B"):
            LinSys(A=valid_A, B=valid_B_2d, w=w_invalid)

    def test_w_wrong_ndim(self, valid_A, valid_B_1d):
        """Test error when w has wrong number of dimensions."""
        w_invalid = torch.tensor([[0.5], [1.5]])
        with pytest.raises(ValueError, match="w must have the same shape as B"):
            LinSys(A=valid_A, B=valid_B_1d, w=w_invalid)


class TestLinSysComputeResidualNorm:
    """Test compute_residual_norm method of LinSys."""

    def test_residual_norm_with_regularization(self, valid_A, valid_B_1d):
        """Test residual norm with regularization."""
        reg = 0.5
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=reg)
        # Solve (A + reg*I)v = B
        A_reg = valid_A + reg * torch.eye(2)
        v_exact = torch.linalg.solve(A_reg, valid_B_1d)
        res_norm = lin_sys.compute_residual_norm(v_exact)
        assert torch.allclose(res_norm, torch.tensor(0.0), atol=1e-6)

    def test_relative_residual_norm(self, valid_A, valid_B_1d):
        """Test relative residual norm."""
        lin_sys = LinSys(A=valid_A, B=valid_B_1d, reg=0.0)
        v = torch.tensor([0.0, 0.0])
        rel_res_norm = lin_sys.compute_residual_norm(v, relative=True)
        # Relative residual should be ||Av - B|| / ||B|| = ||B|| / ||B|| = 1.0
        assert torch.allclose(rel_res_norm, torch.tensor(1.0))

    def test_relative_residual_norm_2d_B(self, valid_A, valid_B_2d):
        """Test relative residual norm with 2D B matrix."""
        lin_sys = LinSys(A=valid_A, B=valid_B_2d, reg=0.0)
        v = torch.zeros_like(valid_B_2d)
        rel_res_norm = lin_sys.compute_residual_norm(v, relative=True)
        # Should have one relative residual norm per column, all equal to 1.0
        assert rel_res_norm.shape == (valid_B_2d.shape[1],)
        assert torch.allclose(rel_res_norm, torch.ones(valid_B_2d.shape[1]))
