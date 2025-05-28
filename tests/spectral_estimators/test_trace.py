import pytest
import torch

from rlaopt.linops import SymmetricLinOp
from rlaopt.spectral_estimators.trace import hutchinson, hutch_plus_plus


def make_linop_from_matrix(M: torch.Tensor) -> SymmetricLinOp:
    """
    Wrap a dense symmetric matrix M into a SymmetricLinOp via
    its matvec.
    """

    def matvec(x: torch.Tensor) -> torch.Tensor:
        return M @ x

    return SymmetricLinOp(M.device, M.shape, matvec, dtype=M.dtype)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
@pytest.mark.parametrize("sketch", ["rademacher", "gauss"])
@pytest.mark.parametrize("dim", [1, 3, 5])
def test_hutchinson_zero_matrix(sketch, dim, dtype):
    M = torch.zeros(dim, dim, dtype=dtype)
    A = make_linop_from_matrix(M)
    est = hutchinson(A, k=10, sketch=sketch)
    assert pytest.approx(0.0, abs=1e-4) == est


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
@pytest.mark.parametrize("dim,scale", [(3, 2.3), (4, -5.0), (7, 3.14)])
def test_hutchinson_identity_rademacher_exact(dim, scale, dtype):
    # For Rademacher and A = scale * I, each v^T A v = scale * dim exactly,
    # so the estimator is deterministic for ANY k.
    M = torch.eye(dim, dtype=dtype) * scale
    A = make_linop_from_matrix(M)
    for k in [1, 5, 10, 100]:
        est = hutchinson(A, k=k, sketch="rademacher")
        assert est == pytest.approx(scale * dim, abs=1e-4)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
@pytest.mark.parametrize("sketch", ["rademacher", "gauss"])
@pytest.mark.parametrize("dim", [3, 4, 6])
def test_hutchpp_zero_matrix(sketch, dim, dtype):
    M = torch.zeros(dim, dim, dtype=dtype)
    A = make_linop_from_matrix(M)
    est = hutch_plus_plus(A, k=8, sketch=sketch)
    assert pytest.approx(0.0, abs=1e-4) == est


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_hutchpp_identity_rademacher_exact(dtype):
    # Hutch++ should also give exact result on scale * I with Rademacher
    # with k/2 >= n due to QR decomposition
    dim, scale = 5, -2.5
    M = torch.eye(dim, dtype=dtype) * scale
    A = make_linop_from_matrix(M)
    est = hutch_plus_plus(A, k=dim * 3, sketch="rademacher")
    assert est == pytest.approx(scale * dim, abs=1e-4)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_reproducibility_given_seed(dtype):
    # Fix a random symmetric matrix
    dim = 7
    M = torch.randn(dim, dim, dtype=dtype)
    M = (M + M.t()) * 0.5
    A = make_linop_from_matrix(M)

    torch.manual_seed(1234)
    e1 = hutchinson(A, k=20, sketch="rademacher")
    torch.manual_seed(1234)
    e2 = hutchinson(A, k=20, sketch="rademacher")
    assert e1 == e2

    torch.manual_seed(999)
    p1 = hutch_plus_plus(A, k=30, sketch="rademacher")
    torch.manual_seed(999)
    p2 = hutch_plus_plus(A, k=30, sketch="rademacher")
    assert p1 == p2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sketch", ["rademacher", "gauss"])
def test_hutchpp_both_sdim_gdim_zero(dtype, sketch):
    # k=0 --> s_dim = 0, g_dim = 0
    M = torch.eye(2, dtype=dtype)
    A = make_linop_from_matrix(M)
    trace = hutch_plus_plus(A, k=0, sketch=sketch, sketch_fraction=0.4)
    assert trace == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sketch_fraction", [0.0, 0.01])
def test_hutchpp_only_gdim_active(dtype, sketch_fraction):
    # s_dim == 0, g_dim == k (e.g. k = 3, sketch_fraction=0.01)
    M = torch.eye(3, dtype=dtype)
    A = make_linop_from_matrix(M)
    trace = hutch_plus_plus(
        A, k=3, sketch="rademacher", sketch_fraction=sketch_fraction
    )
    # Should fallback to Hutchinson, so approximately equal to trace(A)
    assert pytest.approx(3.0, rel=0.05) == trace


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sketch", ["rademacher", "gauss"])
def test_hutchpp_only_sdim_active(dtype, sketch):
    # Large sketch_fraction so g_dim == 0 (but < 0.5, the limit)
    k = 12
    sketch_fraction = 0.49
    M = torch.eye(k, dtype=dtype)
    A = make_linop_from_matrix(M)
    trace = hutch_plus_plus(A, k=k, sketch=sketch, sketch_fraction=sketch_fraction)
    # Should fallback to just the sketch component, which should \
    # estimate trace well for the identity
    assert pytest.approx(float(k), rel=0.1) == trace
