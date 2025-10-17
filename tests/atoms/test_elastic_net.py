"""Tests for ElasticNet atom."""

import pytest
import torch

from rlaopt.atoms.elastic_net import ElasticNet
from rlaopt.expression import Variable


@pytest.fixture
def standard_elasticnet():
    """Fixture for standard elastic net with equal L1 and L2 scaling."""
    x = Variable((5,), name="x")
    x.value = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    return ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0), x


@pytest.fixture
def lasso_like_elasticnet():
    """Fixture for lasso-like elastic net (high L1, low L2)."""
    x = Variable((5,), name="x")
    x.value = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    return ElasticNet(x, l1_scaling=1.0, l2_scaling=0.1), x


@pytest.fixture
def ridge_like_elasticnet():
    """Fixture for ridge-like elastic net (low L1, high L2)."""
    x = Variable((5,), name="x")
    x.value = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    return ElasticNet(x, l1_scaling=0.1, l2_scaling=1.0), x


class TestElasticNetInit:
    """Tests for ElasticNet initialization."""

    def test_basic_initialization(self, standard_elasticnet):
        """Test basic initialization with default parameters."""
        elastic, x = standard_elasticnet

        assert elastic is not None
        assert elastic.l1_scaling == 1.0
        assert elastic.l2_scaling == 1.0

    def test_initialization_with_custom_scaling(self):
        """Test initialization with custom scaling factors."""
        x = Variable((10,), name="x")
        elastic = ElasticNet(x, l1_scaling=0.5, l2_scaling=2.0)

        assert elastic.l1_scaling == 0.5
        assert elastic.l2_scaling == 2.0

    @pytest.mark.parametrize(
        "l1_scaling,l2_scaling",
        [
            (0.0, 1.0),  # Pure L2
            (1.0, 0.0),  # Pure L1
            (0.5, 0.5),
            (2.0, 3.0),
            (0.01, 100.0),
        ],
    )
    def test_various_scaling_factors(self, l1_scaling, l2_scaling):
        """Test initialization with various scaling factor combinations."""
        x = Variable((5,), name="x")
        elastic = ElasticNet(x, l1_scaling=l1_scaling, l2_scaling=l2_scaling)

        assert elastic.l1_scaling == l1_scaling
        assert elastic.l2_scaling == l2_scaling

    def test_initialization_with_non_variable_raises_error(self):
        """Test that initialization with non-Variable raises TypeError."""
        not_a_variable = torch.randn(5)

        with pytest.raises(TypeError, match="Expected Variable"):
            ElasticNet(not_a_variable, l1_scaling=1.0, l2_scaling=1.0)

    def test_initialization_with_none_raises_error(self):
        """Test that initialization with None raises TypeError."""
        with pytest.raises(TypeError, match="Expected Variable"):
            ElasticNet(None, l1_scaling=1.0, l2_scaling=1.0)

    def test_default_scaling_factors(self):
        """Test that default scaling factors are 1.0."""
        x = Variable((5,), name="x")
        elastic = ElasticNet(x)

        assert elastic.l1_scaling == 1.0
        assert elastic.l2_scaling == 1.0

    @pytest.mark.parametrize(
        "shape",
        [
            (1,),
            (10,),
            (100,),
            (5, 5),
            (3, 4, 5),
        ],
    )
    def test_various_shapes(self, shape):
        """Test initialization with various variable shapes."""
        x = Variable(shape, name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        assert elastic is not None


class TestElasticNetForward:
    """Tests for forward evaluation."""

    def test_forward_standard(self, standard_elasticnet):
        """Test forward with standard elastic net."""
        elastic, x = standard_elasticnet

        result = elastic.forward()

        # Manually compute: l1_norm = 15, l2_norm = 55
        # 1.0 * 15 + (1.0 / 2) * 55 = 15 + 27.5 = 42.5
        expected = torch.tensor(42.5)
        assert torch.allclose(result, expected)

    def test_forward_zero_vector(self):
        """Test forward with zero vector."""
        x = Variable((5,), name="x")
        x.value = torch.zeros(5)
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        result = elastic.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_ones_vector(self):
        """Test forward with ones vector."""
        x = Variable((5,), name="x")
        x.value = torch.ones(5)
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        result = elastic.forward()

        # l1_norm = 5, l2_norm = 5
        # 1.0 * 5 + (1.0 / 2) * 5 = 5 + 2.5 = 7.5
        expected = torch.tensor(7.5)
        assert torch.allclose(result, expected)

    def test_forward_pure_l1(self):
        """Test forward with pure L1 (l2_scaling = 0)."""
        x = Variable((4,), name="x")
        x.value = torch.tensor([1.0, -2.0, 3.0, -4.0])
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=0.0)

        result = elastic.forward()

        # l1_norm = 10, l2_norm contribution = 0
        expected = torch.tensor(10.0)
        assert torch.allclose(result, expected)

    def test_forward_pure_l2(self):
        """Test forward with pure L2 (l1_scaling = 0)."""
        x = Variable((4,), name="x")
        x.value = torch.tensor([1.0, -2.0, 3.0, -4.0])
        elastic = ElasticNet(x, l1_scaling=0.0, l2_scaling=1.0)

        result = elastic.forward()

        # l1_norm contribution = 0, l2_norm = 30
        # (1.0 / 2) * 30 = 15
        expected = torch.tensor(15.0)
        assert torch.allclose(result, expected)

    def test_forward_lasso_like(self, lasso_like_elasticnet):
        """Test forward with lasso-like configuration."""
        elastic, x = lasso_like_elasticnet

        result = elastic.forward()

        # l1_norm = 15, l2_norm = 55
        # 1.0 * 15 + (0.1 / 2) * 55 = 15 + 2.75 = 17.75
        expected = torch.tensor(17.75)
        assert torch.allclose(result, expected)

    def test_forward_ridge_like(self, ridge_like_elasticnet):
        """Test forward with ridge-like configuration."""
        elastic, x = ridge_like_elasticnet

        result = elastic.forward()

        # l1_norm = 15, l2_norm = 55
        # 0.1 * 15 + (1.0 / 2) * 55 = 1.5 + 27.5 = 29.0
        expected = torch.tensor(29.0)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("seed", range(5))
    def test_forward_random(self, seed):
        """Test forward with random values."""
        torch.manual_seed(seed)

        x = Variable((10,), name="x")
        x.value = torch.randn(10)
        elastic = ElasticNet(x, l1_scaling=0.5, l2_scaling=0.5)

        result = elastic.forward()

        # Manually compute expected
        l1_norm = torch.sum(torch.abs(x.value))
        l2_norm = torch.sum(x.value**2)
        expected = 0.5 * l1_norm + (0.5 / 2) * l2_norm

        assert torch.allclose(result, expected)

    def test_forward_matrix(self):
        """Test forward with matrix-shaped variable."""
        x = Variable((3, 4), name="x")
        x.value = torch.randn(3, 4)
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        result = elastic.forward()

        l1_norm = torch.sum(torch.abs(x.value))
        l2_norm = torch.sum(x.value**2)
        expected = 1.0 * l1_norm + (1.0 / 2) * l2_norm

        assert torch.allclose(result, expected)

    def test_forward_large_scale(self):
        """Test forward with large-scale variable."""
        x = Variable((1000,), name="x")
        x.value = torch.randn(1000)
        elastic = ElasticNet(x, l1_scaling=0.1, l2_scaling=0.1)

        result = elastic.forward()

        l1_norm = torch.sum(torch.abs(x.value))
        l2_norm = torch.sum(x.value**2)
        expected = 0.1 * l1_norm + (0.1 / 2) * l2_norm

        assert torch.allclose(result, expected)


class TestElasticNetProperties:
    """Tests for elastic net properties."""

    def test_is_smooth_always_false(self, standard_elasticnet):
        """Test that is_smooth always returns False."""
        elastic, _ = standard_elasticnet

        assert elastic.is_smooth() is False

    def test_is_proxable_always_true(self, standard_elasticnet):
        """Test that is_proxable always returns True."""
        elastic, _ = standard_elasticnet

        assert elastic.is_proxable() is True

    def test_is_subsamplable_always_false(self, standard_elasticnet):
        """Test that is_subsamplable always returns False."""
        elastic, _ = standard_elasticnet

        assert elastic.is_subsamplable() is False

    @pytest.mark.parametrize(
        "l1_scaling,l2_scaling",
        [
            (0.0, 1.0),
            (1.0, 0.0),
            (0.5, 0.5),
            (2.0, 3.0),
        ],
    )
    def test_properties_consistent(self, l1_scaling, l2_scaling):
        """Test that properties are consistent across different scalings."""
        x = Variable((5,), name="x")
        elastic = ElasticNet(x, l1_scaling=l1_scaling, l2_scaling=l2_scaling)

        assert elastic.is_smooth() is False
        assert elastic.is_proxable() is True
        assert elastic.is_subsamplable() is False


class TestElasticNetProx:
    """Tests for proximal operator."""

    def test_prox_standard_positive_values(self):
        """Test prox with positive values."""
        x = Variable((3,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([3.0, 2.0, 1.0])
        prox_scaling = 1.0
        projected = elastic.prox(location, prox_scaling)

        # threshold = 1.0 * 1.0 = 1.0
        # l2_term = 1 + 1.0 * 1.0 = 2.0
        # soft_threshold then divide by 2.0
        # [3-1, 2-1, 1-1] / 2.0 = [2, 1, 0] / 2.0 = [1, 0.5, 0]
        expected = torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(projected, expected)

    def test_prox_standard_negative_values(self):
        """Test prox with negative values."""
        x = Variable((3,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([-3.0, -2.0, -1.0])
        prox_scaling = 1.0
        projected = elastic.prox(location, prox_scaling)

        # threshold = 1.0
        # l2_term = 2.0
        # soft_threshold: [-3+1, -2+1, -1+1] / 2.0 = [-2, -1, 0] / 2.0
        expected = torch.tensor([-1.0, -0.5, 0.0])
        assert torch.allclose(projected, expected)

    def test_prox_mixed_values(self):
        """Test prox with mixed positive and negative values."""
        x = Variable((5,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([3.0, -2.0, 1.0, -1.0, 0.0])
        prox_scaling = 1.0
        projected = elastic.prox(location, prox_scaling)

        expected = torch.tensor([1.0, -0.5, 0.0, 0.0, 0.0])
        assert torch.allclose(projected, expected)

    def test_prox_zero_location(self):
        """Test prox with zero location."""
        x = Variable((5,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.zeros(5)
        projected = elastic.prox(location, prox_scaling=1.0)

        # Should remain zero
        assert torch.allclose(projected, torch.zeros(5))

    def test_prox_pure_l1(self):
        """Test prox with pure L1 (l2_scaling = 0)."""
        x = Variable((3,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=0.0)

        location = torch.tensor([3.0, 2.0, 1.0])
        prox_scaling = 1.0
        projected = elastic.prox(location, prox_scaling)

        # threshold = 1.0, l2_term = 1.0
        # Standard soft-thresholding
        expected = torch.tensor([2.0, 1.0, 0.0])
        assert torch.allclose(projected, expected)

    def test_prox_pure_l2(self):
        """Test prox with pure L2 (l1_scaling = 0)."""
        x = Variable((3,), name="x")
        elastic = ElasticNet(x, l1_scaling=0.0, l2_scaling=1.0)

        location = torch.tensor([3.0, 2.0, 1.0])
        prox_scaling = 1.0
        projected = elastic.prox(location, prox_scaling)

        # threshold = 0, l2_term = 2.0
        # Just scaling by 1/2
        expected = torch.tensor([1.5, 1.0, 0.5])
        assert torch.allclose(projected, expected)

    @pytest.mark.parametrize("prox_scaling", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_prox_various_scaling(self, prox_scaling):
        """Test prox with various prox_scaling values."""
        x = Variable((3,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([5.0, 3.0, 1.0])
        projected = elastic.prox(location, prox_scaling)

        # Verify the formula
        l2_term = 1 + prox_scaling * 1.0
        threshold = 1.0 * prox_scaling
        expected = (
            torch.nn.functional.relu(location - threshold)
            - torch.nn.functional.relu(-location - threshold)
        ) / l2_term

        assert torch.allclose(projected, expected)

    def test_prox_small_threshold(self):
        """Test prox with small threshold (small l1_scaling or prox_scaling)."""
        x = Variable((3,), name="x")
        elastic = ElasticNet(x, l1_scaling=0.1, l2_scaling=0.1)

        location = torch.tensor([1.0, 0.5, 0.1])
        prox_scaling = 0.1
        projected = elastic.prox(location, prox_scaling)

        # threshold = 0.1 * 0.1 = 0.01
        # l2_term = 1 + 0.1 * 0.1 = 1.01
        l2_term = 1.01
        threshold = 0.01
        expected = (
            torch.nn.functional.relu(location - threshold)
            - torch.nn.functional.relu(-location - threshold)
        ) / l2_term

        assert torch.allclose(projected, expected, atol=1e-6)

    def test_prox_large_values(self):
        """Test prox with large values."""
        x = Variable((3,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([100.0, -100.0, 50.0])
        prox_scaling = 1.0
        projected = elastic.prox(location, prox_scaling)

        # threshold = 1.0, l2_term = 2.0
        expected = torch.tensor([49.5, -49.5, 24.5])
        assert torch.allclose(projected, expected)

    def test_prox_shrinkage_property(self):
        """Test that prox shrinks values towards zero."""
        x = Variable((5,), name="x")
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        projected = elastic.prox(location, prox_scaling=1.0)

        # All values should be shrunk towards zero
        assert torch.all(torch.abs(projected) < torch.abs(location))


class TestElasticNetSubsample:
    """Tests for subsampling (should raise error)."""

    def test_subsample_raises_not_implemented(self, standard_elasticnet):
        """Test that subsample raises NotImplementedError."""
        elastic, _ = standard_elasticnet

        with pytest.raises(NotImplementedError, match="does not support subsampling"):
            elastic.subsample(torch.tensor([0, 1, 2]))


class TestElasticNetEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype):
        """Test with different floating point precisions."""
        x = Variable((5,), name="x")
        x.value = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=dtype)
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        result = elastic.forward()
        assert result.dtype == dtype

        location = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=dtype)
        projected = elastic.prox(location, prox_scaling=1.0)
        assert projected.dtype == dtype

    def test_very_small_values(self):
        """Test with very small values."""
        x = Variable((3,), name="x")
        x.value = torch.tensor([1e-10, -1e-10, 1e-10], dtype=torch.float64)
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        result = elastic.forward()

        # Should handle small values correctly
        assert result >= 0
        assert result < 1e-8

    def test_very_large_scaling(self):
        """Test with very large scaling factors."""
        x = Variable((3,), name="x")
        x.value = torch.tensor([1.0, 2.0, 3.0])
        elastic = ElasticNet(x, l1_scaling=1e6, l2_scaling=1e6)

        result = elastic.forward()

        # Should handle large scaling factors
        assert torch.isfinite(result)

    def test_zero_scaling_factors(self):
        """Test with zero scaling factors."""
        x = Variable((3,), name="x")
        x.value = torch.tensor([1.0, 2.0, 3.0])
        elastic = ElasticNet(x, l1_scaling=0.0, l2_scaling=0.0)

        result = elastic.forward()

        # Should be zero when both scaling factors are zero
        assert torch.allclose(result, torch.tensor(0.0))

    def test_negative_values_in_forward(self):
        """Test that forward handles negative values correctly (via abs)."""
        x = Variable((4,), name="x")
        x.value = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        elastic = ElasticNet(x, l1_scaling=1.0, l2_scaling=1.0)

        result = elastic.forward()

        # Should be same as positive values due to abs in L1
        x2 = Variable((4,), name="x2")
        x2.value = torch.tensor([1.0, 2.0, 3.0, 4.0])
        elastic2 = ElasticNet(x2, l1_scaling=1.0, l2_scaling=1.0)
        result2 = elastic2.forward()

        assert torch.allclose(result, result2)

    def test_3d_tensor(self):
        """Test with 3D tensor variable."""
        x = Variable((2, 3, 4), name="x")
        x.value = torch.randn(2, 3, 4)
        elastic = ElasticNet(x, l1_scaling=0.5, l2_scaling=0.5)

        result = elastic.forward()

        l1_norm = torch.sum(torch.abs(x.value))
        l2_norm = torch.sum(x.value**2)
        expected = 0.5 * l1_norm + (0.5 / 2) * l2_norm

        assert torch.allclose(result, expected)


class TestElasticNetExamples:
    """Tests based on docstring examples."""

    def test_docstring_standard_example(self):
        """Test the standard elastic net example from docstring."""
        x = Variable((100,), name="weights")
        elastic = ElasticNet(x, l1_scaling=0.5, l2_scaling=0.5)

        x.value = torch.randn(100)
        penalty = elastic.forward()

        assert penalty >= 0
        assert torch.isfinite(penalty)

    def test_docstring_lasso_like_example(self):
        """Test the lasso-like example from docstring."""
        x = Variable((100,), name="weights")
        elastic_lasso = ElasticNet(x, l1_scaling=1.0, l2_scaling=0.1)

        assert elastic_lasso.l1_scaling == 1.0
        assert elastic_lasso.l2_scaling == 0.1

    def test_docstring_ridge_like_example(self):
        """Test the ridge-like example from docstring."""
        x = Variable((100,), name="weights")
        elastic_ridge = ElasticNet(x, l1_scaling=0.1, l2_scaling=1.0)

        assert elastic_ridge.l1_scaling == 0.1
        assert elastic_ridge.l2_scaling == 1.0


@pytest.mark.parametrize(
    "l1_scaling,l2_scaling,seed",
    [
        (1.0, 1.0, 0),
        (0.5, 0.5, 1),
        (1.0, 0.1, 2),
        (0.1, 1.0, 3),
        (0.0, 1.0, 4),
        (1.0, 0.0, 5),
    ],
)
def test_elasticnet_general(l1_scaling, l2_scaling, seed):
    """General test for ElasticNet with various configurations."""
    torch.manual_seed(seed)

    x = Variable((20,), name="x")
    x.value = torch.randn(20)
    elastic = ElasticNet(x, l1_scaling=l1_scaling, l2_scaling=l2_scaling)

    # Test forward
    result = elastic.forward()
    l1_norm = torch.sum(torch.abs(x.value))
    l2_norm = torch.sum(x.value**2)
    expected = l1_scaling * l1_norm + (l2_scaling / 2) * l2_norm
    assert torch.allclose(result, expected)

    # Test properties
    assert elastic.is_smooth() is False
    assert elastic.is_proxable() is True
    assert elastic.is_subsamplable() is False

    # Test prox
    location = torch.randn(20)
    projected = elastic.prox(location, prox_scaling=1.0)

    # Verify prox formula
    l2_term = 1 + 1.0 * l2_scaling
    threshold = l1_scaling * 1.0
    expected_prox = (
        torch.nn.functional.relu(location - threshold)
        - torch.nn.functional.relu(-location - threshold)
    ) / l2_term
    assert torch.allclose(projected, expected_prox)
