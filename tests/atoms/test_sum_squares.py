import pytest
import torch

from rlaopt.atoms.sum_squares import SumSquares
from rlaopt.expression.expression import Variable

TOLERANCES = {torch.float32: 1e-5, torch.float64: 1e-10}


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    return request.param


@pytest.fixture
def tol(precision):
    return TOLERANCES[precision]


class TestSumSquaresVariable:
    @pytest.fixture(autouse=True)
    def _setup_test(self, precision):
        # Set seed per test for isolation
        torch.manual_seed(0)

        x0 = torch.randn(64, dtype=precision)
        x = Variable(x0, name="x")
        self.f = SumSquares(x)
        self.x0 = x0

    def test_init__(self):
        assert self.f.x is not None
        assert isinstance(self.f.x, torch.nn.Parameter)

    def test_bool_methods(self):
        assert self.f.is_smooth() is True
        assert self.f.is_proxable() is True

    def test_forward(self, tol):
        ans = (self.x0**2).sum()
        assert torch.linalg.norm(self.f.forward() - ans) <= tol

    def test_prox(self, precision, tol):
        v = torch.randn(64, dtype=precision)
        prox = self.f.prox(v, 1.0)
        assert torch.linalg.norm(prox - v / 2.0) <= tol
