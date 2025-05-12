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


@pytest.mark.parametrize("sketch", ["rademacher", "gauss"])
@pytest.mark.parametrize("dim", [1, 3, 5])
def test_hutchinson_zero_matrix(sketch, dim):
    M = torch.zeros(dim, dim, dtype=torch.float32)
    A = make_linop_from_matrix(M)
    est = hutchinson(A, k=10, sketch=sketch)
    assert pytest.approx(0.0, abs=1e-4) == est


@pytest.mark.parametrize("dim,scale", [(1, 2.3), (4, -5.0), (7, 3.14)])
def test_hutchinson_identity_rademacher_exact(dim, scale):
    # For Rademacher and A = scale * I, each v^T A v = scale * dim exactly,
    # so the estimator is deterministic for ANY k.
    M = torch.eye(dim, dtype=torch.float32) * scale
    A = make_linop_from_matrix(M)
    for k in [1, 5, 10, 100]:
        est = hutchinson(A, k=k, sketch="rademacher")
        assert est == pytest.approx(scale * dim, abs=1e-4)


@pytest.mark.parametrize("sketch", ["rademacher", "gauss"])
@pytest.mark.parametrize("dim", [1, 4, 6])
def test_hutchpp_zero_matrix(sketch, dim):
    M = torch.zeros(dim, dim, dtype=torch.float32)
    A = make_linop_from_matrix(M)
    est = hutch_plus_plus(A, k=8, sketch=sketch)
    assert pytest.approx(0.0, abs=1e-4) == est


def test_hutchpp_identity_rademacher_exact():
    # Hutch++ should also give exact result on scale * I with Rademacher
    # with k/2 >= n due to QR decomposition
    dim, scale = 5, -2.5
    M = torch.eye(dim, dtype=torch.float32) * scale
    A = make_linop_from_matrix(M)
    est = hutch_plus_plus(A, k=dim * 2, sketch="rademacher")
    assert est == pytest.approx(scale * dim, abs=1e-4)


def test_reproducibility_given_seed():
    # Fix a random symmetric matrix
    dim = 7
    M = torch.randn(dim, dim, dtype=torch.float32)
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
