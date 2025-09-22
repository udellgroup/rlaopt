import pytest
import torch

from rlaopt.atoms.sum_squares import SumSquares
from rlaopt.atoms.non_negative import NonNegative
from rlaopt.expression.expression import Variable
from rlaopt.solvers.configs import ProxGradConfig
from rlaopt.solvers.proximal_gradient.prox_grad import ProximalGradient


torch.manual_seed(0)

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
def tol(precision):
    return TOLERANCES[precision]

class TestNonNegativeBasics:
    def test_init(self, test_var):
        r = NonNegative(test_var)
        assert r.x is not None
    
    def test_forward(self, test_var, test_dim, precision):
        r = NonNegative(test_var)
        assert r.forward() == 0.0

        params = {"x": -torch.ones(test_dim, dtype=precision)}
        assert r.evaluate(params) == torch.inf

    def test_prox(self, test_var, test_dim, precision):
        r = NonNegative(test_var)
        
        v, zero = torch.ones(test_dim, dtype=precision), torch.zeros(test_dim, dtype=precision)
        scaling_factor = 1.0
        
        prox = r.prox(-v, scaling_factor)
        assert (prox == zero).all() == True

        prox = r.prox(v, scaling_factor)
        assert (prox == v).all() == True
    

class TestNonNegativeSolve:
    
    def test_prox_grad(self, precision, tol):
        torch.set_default_dtype(precision)
        
        n, p = 1024, 256
        A = torch.randn((n, p)) / n ** (0.5)
        b = torch.randn(n) / n ** (0.5)
        x = Variable(torch.zeros(p))
        
        obj = SumSquares(A @ x - b) + NonNegative(x)
        
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
        
        # Params are non-negative
        assert (params >= 0).all() == True

        # Problem is solved
        assert (err.item() <= tol) == True


        

