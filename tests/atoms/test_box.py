import pytest
import torch

from rlaopt.atoms.non_negative import Box
from rlaopt.atoms.sum_squares import SumSquares
from rlaopt.expression.expression import Variable
from rlaopt.solvers.proximal_gradient.prox_grad import ProxGradConfig, ProximalGradient

TOLERANCES = {torch.float32: 1e-6, torch.float64: 1e-10}


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    return request.param


@pytest.fixture
def test_dim():
    return 128


@pytest.fixture
def test_var(test_dim, precision):
    x = Variable(torch.zeros(test_dim, dtype=precision))
    return x


@pytest.fixture
def l():
    return torch.tensor(-2.0)


@pytest.fixture
def u():
    return torch.tensor(1.0)


@pytest.fixture
def tol(precision):
    return TOLERANCES[precision]


@pytest.fixture
def reset_torch_state():
    """Fixture to reset torch default dtype after each test"""
    original_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(original_dtype)


class TestBoxBasics:
    def test_init(self, test_var, l, u):
        torch.manual_seed(0)
        r = Box(test_var, l=l, u=u)
        assert r.x is not None

    def test_forward(self, test_var, l, u, test_dim, precision):
        torch.manual_seed(0)
        r = Box(test_var, l=l, u=u)
        assert r.forward() == 0.0

        params = {"x": -3 * torch.ones(test_dim, dtype=precision)}
        assert r.evaluate(params) == torch.inf

    def test_prox(self, test_var, l, u, test_dim, precision):
        torch.manual_seed(0)
        r = Box(test_var, l=l, u=u)

        v = torch.ones(test_dim, dtype=precision)
        scaling_factor = 1.0

        prox = r.prox(-v, scaling_factor)
        assert (prox == -v).all() == True

        prox = r.prox(-4 * v, scaling_factor)
        assert (prox == -2.0 * v).all() == True


class TestBoxSolve:
    def test_prox_grad(self, reset_torch_state, l, u, precision, tol):
        torch.manual_seed(0)
        torch.set_default_dtype(precision)

        n, p = 1024, 256
        A = torch.randn((n, p)) / n ** (0.5)
        b = torch.randn(n) / n ** (0.5)
        x = Variable(torch.zeros(p))

        obj = SumSquares(A @ x - b) + Box(x, l=l, u=u)

        eta = 0.5 / torch.linalg.norm(A, ord=2) ** 2

        config = ProxGradConfig(eta=eta, use_acceleration=True, use_linesearch=False)
        opt = ProximalGradient(config, obj)
        params = obj.params
        state = opt.init_state(params)

        for _ in range(500):
            params, state = opt.step(params, state)
        err = state.err

        names = list(params.keys())
        params = params[names[0]]

        # Params are in the box
        assert (params >= l).all() & (params <= u).all() == True

        # Problem is solved
        assert (err.item() <= tol) == True
