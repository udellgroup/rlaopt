import pytest
import torch

from rlaopt.atoms.affine_constraint import AffineConstraint
from rlaopt.expression.expression import Variable

TOLERANCES = {torch.float32: 1e-5, torch.float64: 1e-10}


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    return request.param


@pytest.fixture
def tol(precision):
    return TOLERANCES[precision]


class TestAffineConstraint:
    @pytest.fixture(autouse=True)
    def _setup_test(self, precision):
        # Set seed per test for isolation
        torch.manual_seed(0)

        A = torch.randn((16, 64), dtype=precision) / 4.0
        x0 = torch.randn(64, dtype=precision)
        b = A @ x0
        x = Variable(torch.zeros(64, dtype=precision), name="x")
        self.r = AffineConstraint(x, A, b)
        self.x0 = x0

    def test_init__(self):
        assert self.r.x is not None
        assert self.r.A is not None
        assert self.r.b is not None

    def test_bool_methods(self):
        assert self.r.is_smooth() is False
        assert self.r.is_proxable() is True
        assert self.r.is_subsamplable() is False

    def test_forward(self):
        assert (self.r.forward() == torch.inf) == True

        params = {"x": self.x0}
        assert (self.r.evaluate(params) == 0.0) == True

    def test_prox(self, precision, tol):
        v = torch.randn(64, dtype=precision)
        prox = self.r.prox(v, 1.0)
        assert torch.linalg.norm(self.r.A @ prox - self.r.b) <= tol

        prox = self.r.prox(self.x0, 1.0)
        assert torch.linalg.norm(prox - self.x0) <= tol
