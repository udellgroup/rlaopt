import pytest
import torch

from rlaopt.atoms.l1norm import L1Norm
from rlaopt.expression.expression import Variable

TOLERANCES = {torch.float32: 1e-6, torch.float64: 1e-10}


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    return request.param


@pytest.fixture
def tol(precision):
    return TOLERANCES[precision]


class TestL1NormBasics:
    @pytest.fixture(autouse=True)
    def _setup_test(self, precision):
        # Set seed per test for isolation
        torch.manual_seed(0)

        self.p = 3

        _x = torch.tensor([100, 50, 10], dtype=precision)
        self._x = _x
        self.x = Variable(_x)
        self.lambd = 2.0
        self.r = L1Norm(self.x, scaling=self.lambd)

    def test_init(self):
        assert self.r.x is not None
        assert self.r.scaling is not None
        assert self.r.is_smooth() == False
        assert self.r.is_proxable() == True
        assert self.r.is_subsamplable() == False

    def test_forward(self):
        assert (
            torch.isclose(
                self.r.forward(), self.lambd * torch.linalg.norm(self._x, ord=1)
            )
            == True
        )

    def test_prox(self, tol):
        scaling_factor = 20.0

        # When lambd = 2.0 and scaling factor is 20,
        # prox should threshold last entry of x to 0,
        # and subtract 40 from all other entries
        prox = self.r.prox(self._x, scaling_factor)

        ans = torch.clone(self._x)
        ans[0:2] -= scaling_factor * self.lambd
        ans[-1] = 0.0

        err = torch.linalg.norm(prox - ans, ord=2)
        assert (err.item() <= tol) == True

        # When lambd = 2.0 and scaling factor is 20,
        # all entries of x should be thresholded to 0.0
        scaling_factor = 200.0
        prox = self.r.prox(self._x, scaling_factor)
        err = torch.linalg.norm(prox, ord=2)
        assert (err.item() <= tol) == True
