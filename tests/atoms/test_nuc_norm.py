import pytest
import torch

from rlaopt.atoms.nuc_norm import NucNorm
from rlaopt.atoms.sum_squares import SumSquares
from rlaopt.expression.expression import Variable
from rlaopt.solvers import ProxGrad, ProxGradConfig

TOLERANCES = {torch.float32: 1e-6, torch.float64: 1e-10}


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    return request.param


@pytest.fixture
def tol(precision):
    return TOLERANCES[precision]


@pytest.fixture
def reset_torch_state():
    """Fixture to reset torch default dtype after each test"""
    original_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(original_dtype)


def get_atol(precision, t: torch.Tensor):
    return torch.linalg.norm(t, ord=2) * TOLERANCES[precision]


class TestNucNormBasics:
    @pytest.fixture(autouse=True)
    def _setup_test(self, precision):
        # Set seed per test for isolation
        torch.manual_seed(0)

        self.p = 5
        self.U, _ = torch.linalg.qr(torch.randn(self.p, self.p, dtype=precision))
        self.V, _ = torch.linalg.qr(torch.randn(self.p, self.p, dtype=precision))
        self.S = torch.tensor([100, 50, 5, 0.1, 0.01], dtype=precision)
        self.x = Variable(self.U @ (self.S * self.V.T))
        self.r = NucNorm(self.x, scaling=1.0)
        self.atol = get_atol(precision, self.S)

    def test_init(self):
        assert self.r.x is not None
        assert self.r.scaling is not None

    def test_forward(self):
        assert torch.isclose(self.r.forward(), torch.sum(self.S)) == True

    def test_prox(self):
        scaling_factor = 1.0

        # When lambd = 1 and scaling factor is 1,
        # prox should threshold last two singular values to 0,
        # and subtract 1 from all other singular values.
        S_new = torch.clone(self.S)
        S_new[0:3] -= 1.0
        S_new[3:5] = 0.0
        ans = self.U @ (S_new * self.V.T)

        prox = self.r.prox(self.x.value.data, scaling_factor)
        err = torch.linalg.norm(prox - ans, ord=2)
        assert (err.item() <= self.atol) == True

        # When scaling factor is 200, all singular values should
        # be thresholded to zero.
        scaling_factor = 200
        prox = self.r.prox(self.x.value.data, scaling_factor)
        err = torch.linalg.norm(prox, ord=2)
        assert (err.item() <= self.atol) == True


class TestNucNormProxGrad:
    def test_prox_grad(self, reset_torch_state):
        torch.manual_seed(0)
        tol = TOLERANCES[torch.float64]
        torch.set_default_dtype(torch.float64)

        X_star = torch.randn(64, 8)
        Y_star = torch.randn(8, 16)
        W_Star = X_star @ Y_star
        A = torch.randn(128, 64)
        B = A @ W_Star + 10**-4 * torch.randn((128, 16))
        lambd = 1000.0

        W = Variable(torch.zeros_like(W_Star))
        obj = SumSquares(A @ W - B) + NucNorm(W, scaling=lambd)

        config = ProxGradConfig(
            eta=10**-3, use_acceleration=False, use_linesearch=False
        )
        opt = ProxGrad(obj, config)
        params = obj.params
        state = opt.init_state(params)

        for _ in range(1000):
            params, state = opt.step(params, state)

        names = list(params.keys())
        params = params[names[0]]
        err = state.err.item()

        # Params has rank less than 16
        assert (torch.linalg.matrix_rank(params) < 16) == True

        # Problem is solved
        assert (err <= tol) == True
