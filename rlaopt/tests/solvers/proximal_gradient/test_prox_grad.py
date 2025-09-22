import numpy as np
import pytest
import torch

from rlaopt.atoms import (
  SumSquares, AffineConstraint, Box, L1Norm, NonNegative, NucNorm
  )

from rlaopt.expression.expression import Variable
from rlaopt.solvers.configs import ProxGradConfig
from rlaopt.solvers.proximal_gradient.prox_grad import ProximalGradient

ACCEL = {"accel": True, "no_accel": False}
LINESEARCH = {"linesearch": True, "no_linesarch": False}
TOLERANCES = {torch.float32: 1e-4, torch.float64: 1e-10}

@pytest.fixture(params=["accel", "no_accel"], ids=["accel", "no_accel"])
def accel(request):
  return request.param

@pytest.fixture(params=["linesearch", "no_linesearch"], 
                ids=["linesearch", "no_linesearch"]
)
def linesearch(request):
  return request.params

@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    return request.param

@pytest.fixture
def acceleration(accel):
  return ACCEL[accel]

@pytest.fixture
def ls(linesearch):
  return LINESEARCH[linesearch]

@pytest.fixture
def tol(precision):
    return TOLERANCES[precision]

class TestProxGrad:
    @pytest.fixture(autouse=True)
    def test_fn(self, obj, eta, tol, acceleration, ls):
        return _solve_test_fn(obj, eta, tol, acceleration, ls)

    def test_least_squares(self, precision, tol):
        torch.set_default_dtype(precision)

        A = torch.randn(1024, 256) / (1024) ** (0.5)
        b = torch.randn(1024) / (1024) ** (0.5)
        x = Variable(torch.zeros(1024))

        obj = SumSquares(A @ x - b)
        eta = 0.5 / torch.linalg.norm(A, ord=2) ** 2
        
        self.test_fn(obj, eta, tol)
    
    def test_lasso(self, precision, tol):
      torch.set_default_dtype(precision)

      n, p = 1024, 128
      s = 32 
      J = np.random.choice(p, s)
      xStar = torch.zeros(p)
      xStar[J] = torch.randn(s) / (s ** 0.5)
      A = torch.randn(n, p) / (n ** 0.5)
      b = A @ xStar + 0.001 * torch.randn(n)

      x = Variable(torch.zeros(p))
      mu = 0.1 * torch.linalg.norm(A.T @ b, ord=torch.inf)
      obj = SumSquares(A @ x - b) + L1Norm(x, scaling=mu) 
      eta = 0.5 / torch.linalg.norm(A, ord=2) ** 2
        
      self.test_fn(obj, eta, tol)

def _solve_test_fn(
    obj, 
    eta, 
    tol,
    use_acceleration,
    use_linesearch
):
  
  opt = _build_opt(obj, eta, tol, use_acceleration, use_linesearch)

  params, state = _init_opt(obj, opt)

  # Test solving by step
  params, err = _loop(params, state, opt)
  assert (err <= tol)

  # Test using solve method
  params, err = opt.solve(obj)
  assert (err.item() <= tol)

  
def _loop(params, state, opt):
 for _ in range(opt.config.max_iters):
  params, state = opt.step(params, state)
  if state.err.item() <= opt.config.tol:
   break
  return params, state.err.item()


def _init_opt(obj, opt):
 params = obj.params
 return params, opt.init_state(params)

def _build_opt(obj, eta, tol, use_acceleration, use_linesearch):
 config = ProxGradConfig(
 eta=eta,
 tol=tol, 
 use_acceleration=use_acceleration,
 use_linesearch=use_linesearch
)
 return ProximalGradient(config, obj) 