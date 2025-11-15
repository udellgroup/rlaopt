"""Tests for ext_tensordict."""

import pytest
import torch

from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def simple_td():
    """Create a simple TensorDict for testing."""
    return TensorDict(
        {"a": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([4.0, 5.0])},
        batch_size=[],
    )


@pytest.fixture
def simple_td_2():
    """Create a second simple TensorDict with same structure."""
    return TensorDict(
        {"a": torch.tensor([2.0, 3.0, 4.0]), "b": torch.tensor([1.0, 2.0])},
        batch_size=[],
    )


@pytest.fixture
def batched_td():
    """Create a TensorDict with batch dimensions."""
    return TensorDict({"x": torch.randn(3, 5), "y": torch.randn(3, 2)}, batch_size=[3])


class TestTensorDictMethods:
    """Test custom TensorDict methods."""

    def test_dim_f(self, simple_td):
        """Test total dimension calculation."""
        assert simple_td.flat_dim() == 5  # 3 + 2 elements

    def test_dim_f_batched(self, batched_td):
        """Test dimension calculation with batched tensors."""
        assert batched_td.flat_dim() == 21  # 3*5 + 3*2 = 15 + 6

    def test_dot_f(self, simple_td, simple_td_2):
        """Test flat dot product."""
        result = simple_td.flat_dot(simple_td_2)
        # (1*2 + 2*3 + 3*4) + (4*1 + 5*2) = (2+6+12) + (4+10) = 20 + 14 = 34
        assert result.item() == pytest.approx(34.0)

    def test_dot_f_self(self, simple_td):
        """Test dot product with itself."""
        result = simple_td.flat_dot(simple_td)
        # 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55
        assert result.item() == pytest.approx(55.0)

    def test_norm_f(self, simple_td):
        """Test L2 norm calculation."""
        result = simple_td.flat_norm()
        expected = torch.sqrt(torch.tensor(55.0))  # sqrt(1+4+9+16+25)
        assert result.item() == pytest.approx(expected.item())

    def test_to_flat_tensor(self, simple_td):
        """Test flattening to 1D tensor."""
        vec = simple_td.to_flat_tensor()
        assert vec.shape == (5,)
        assert torch.allclose(vec, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_to_flat_tensor_empty(self):
        """Test to_flat_tensor on empty TensorDict."""
        empty_td = TensorDict({}, batch_size=[])
        vec = empty_td.to_flat_tensor()
        assert vec.shape == (0,)

    def test_from_flat_tensor(self, simple_td):
        """Test reconstruction from flat_tensor."""
        flat_tensor = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        reconstructed = simple_td.from_flat_tensor(flat_tensor)

        assert torch.allclose(reconstructed["a"], torch.tensor([10.0, 20.0, 30.0]))
        assert torch.allclose(reconstructed["b"], torch.tensor([40.0, 50.0]))
        assert reconstructed.batch_size == simple_td.batch_size

    def test_to_flat_tensor_from_flat_tensor_roundtrip(self, simple_td):
        """Test that to_flat_tensor and from_vector are inverses."""
        flat_tensor = simple_td.to_flat_tensor()
        reconstructed = simple_td.from_flat_tensor(flat_tensor)

        for key in simple_td.keys():
            assert torch.allclose(simple_td[key], reconstructed[key])


class TestPyTreeIntegration:
    """Test PyTree registration for torch.func compatibility."""

    def test_grad_compatibility(self):
        """Test that TensorDict works with torch.func.grad."""

        def loss_fn(params):
            return (params["a"] ** 2).sum() + (params["b"] ** 2).sum()

        # Create TensorDict with requires_grad
        td = TensorDict(
            {
                "a": torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                "b": torch.tensor([4.0, 5.0], requires_grad=True),
            },
            batch_size=[],
        )

        grads = torch.func.grad(loss_fn)(td)

        # Gradients should be 2*x for each element
        assert torch.allclose(grads["a"], torch.tensor([2.0, 4.0, 6.0]))
        assert torch.allclose(grads["b"], torch.tensor([8.0, 10.0]))
