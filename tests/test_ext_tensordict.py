"""Tests for ext_tensordict."""

import pytest
import torch

from rlaopt.ext_tensordict import (
    TensorDict,
    has_compatible_lengths,
    has_compatible_ordered_names,
    has_compatible_ordered_shapes,
    relabel_from_template,
)


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
        assert simple_td.dim_f() == 5  # 3 + 2 elements

    def test_dim_f_batched(self, batched_td):
        """Test dimension calculation with batched tensors."""
        assert batched_td.dim_f() == 21  # 3*5 + 3*2 = 15 + 6

    def test_dot_f(self, simple_td, simple_td_2):
        """Test flat dot product."""
        result = simple_td.dot_f(simple_td_2)
        # (1*2 + 2*3 + 3*4) + (4*1 + 5*2) = (2+6+12) + (4+10) = 20 + 14 = 34
        assert result.item() == pytest.approx(34.0)

    def test_dot_f_self(self, simple_td):
        """Test dot product with itself."""
        result = simple_td.dot_f(simple_td)
        # 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55
        assert result.item() == pytest.approx(55.0)

    def test_norm_f(self, simple_td):
        """Test L2 norm calculation."""
        result = simple_td.norm_f()
        expected = torch.sqrt(torch.tensor(55.0))  # sqrt(1+4+9+16+25)
        assert result.item() == pytest.approx(expected.item())

    def test_to_vector(self, simple_td):
        """Test flattening to vector."""
        vec = simple_td.to_vector()
        assert vec.shape == (5,)
        assert torch.allclose(vec, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_to_vector_empty(self):
        """Test to_vector on empty TensorDict."""
        empty_td = TensorDict({}, batch_size=[])
        vec = empty_td.to_vector()
        assert vec.shape == (0,)

    def test_from_vector(self, simple_td):
        """Test reconstruction from vector."""
        vec = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        reconstructed = simple_td.from_vector(vec)

        assert torch.allclose(reconstructed["a"], torch.tensor([10.0, 20.0, 30.0]))
        assert torch.allclose(reconstructed["b"], torch.tensor([40.0, 50.0]))
        assert reconstructed.batch_size == simple_td.batch_size

    def test_to_vector_from_vector_roundtrip(self, simple_td):
        """Test that to_vector and from_vector are inverses."""
        vec = simple_td.to_vector()
        reconstructed = simple_td.from_vector(vec)

        for key in simple_td.keys():
            assert torch.allclose(simple_td[key], reconstructed[key])

    def test_convert_target(self, simple_td):
        """Test relabeling with different keys."""
        target = TensorDict(
            {
                "x": torch.tensor([100.0, 200.0, 300.0]),
                "y": torch.tensor([400.0, 500.0]),
            },
            batch_size=[],
        )

        result = simple_td.convert_target(target)

        # Should have keys from simple_td, values from target
        assert list(result.keys()) == ["a", "b"]
        assert torch.allclose(result["a"], torch.tensor([100.0, 200.0, 300.0]))
        assert torch.allclose(result["b"], torch.tensor([400.0, 500.0]))


class TestCompatibilityFunctions:
    """Test TensorDict compatibility checking functions."""

    def test_has_compatible_lengths_true(self, simple_td, simple_td_2):
        """Test compatible lengths detection."""
        assert has_compatible_lengths(simple_td, simple_td_2)

    def test_has_compatible_lengths_false(self, simple_td):
        """Test incompatible lengths detection."""
        other_td = TensorDict(
            {
                "a": torch.tensor([1.0]),
                "b": torch.tensor([2.0]),
                "c": torch.tensor([3.0]),
            },
            batch_size=[],
        )
        assert not has_compatible_lengths(simple_td, other_td)

    def test_has_compatible_ordered_names_true(self, simple_td, simple_td_2):
        """Test compatible names detection."""
        assert has_compatible_ordered_names(simple_td, simple_td_2)

    def test_has_compatible_ordered_names_false(self, simple_td):
        """Test incompatible names detection."""
        other_td = TensorDict(
            {"x": torch.tensor([1.0, 2.0, 3.0]), "y": torch.tensor([4.0, 5.0])},
            batch_size=[],
        )
        assert not has_compatible_ordered_names(simple_td, other_td)

    def test_has_compatible_ordered_shapes_true(self, simple_td, simple_td_2):
        """Test compatible shapes detection."""
        assert has_compatible_ordered_shapes(simple_td, simple_td_2)

    def test_has_compatible_ordered_shapes_false(self, simple_td):
        """Test incompatible shapes detection."""
        other_td = TensorDict(
            {
                "a": torch.tensor([1.0, 2.0]),  # Different shape
                "b": torch.tensor([4.0, 5.0]),
            },
            batch_size=[],
        )
        assert not has_compatible_ordered_shapes(simple_td, other_td)

    def test_relabel_from_template(self, simple_td):
        """Test relabeling from template."""
        template = TensorDict(
            {"new_a": torch.zeros(3), "new_b": torch.zeros(2)}, batch_size=[]
        )

        result = relabel_from_template(simple_td, template)

        assert list(result.keys()) == ["new_a", "new_b"]
        assert torch.allclose(result["new_a"], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(result["new_b"], torch.tensor([4.0, 5.0]))

    def test_relabel_from_template_incompatible_raises(self, simple_td):
        """Test that relabeling with incompatible shapes raises error."""
        template = TensorDict(
            {
                "x": torch.zeros(5),  # Wrong shape
                "y": torch.zeros(2),
            },
            batch_size=[],
        )

        with pytest.raises(ValueError, match="Incompatible ordered shapes"):
            relabel_from_template(simple_td, template)


class TestPyTreeIntegration:
    """Test PyTree registration for torch.func compatibility."""

    def test_grad_compatibility(self, simple_td):
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
