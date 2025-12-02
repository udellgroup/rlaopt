"""SapphireSplit class for representing finite-sum composite objective functions."""

from functools import partial
from typing import Callable

import torch
from linops import LinearOperator

from rlaopt.atoms.linear_model.linear_model import LinearModel
from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.prox_grad_split import ProxGradSplit


class SapphireSplit(ProxGradSplit):

    def __init__(self, expr: Expression):
        super().__init__(expr)
        if not isinstance(self.f, LinearModel):
            raise ValueError(f"Smooth part f must be of type LinearModel," \
            f"but recieved {type(self.f).__name__}")
    
        
    def loss(
            self, 
            beta_value: TensorDict, 
            X_batch: torch.Tensor,
            y_batch: torch.Tensor
    )->torch.Tensor:
        return self.f.loss(beta_value, X_batch, y_batch)
    
    def batch_loss_grad(
            self,
            beta_value: TensorDict,
            X_batch: torch.Tensor,
            y_batch: torch.Tensor
    )->TensorDict:
        return torch.func.grad(self.loss)(beta_value, X_batch, y_batch)
    
    def get_subsamp_hessian_linop(
            self, beta_value: TensorDict, X_batch: torch.Tensor, 
            y_batch: torch.Tensor, device: torch.device
    ):
        return _SubampHessianLinOp(self.loss, beta_value, X_batch, y_batch, device)
    
    
class _SubampHessianLinOp(LinearOperator):
    def __init__(self, 
                 loss: Callable[[TensorDict, tuple[torch.Tensor, torch.Tensor]],
                                torch.Tensor], 
                 variable_values: TensorDict, 
                 X_batch: torch.Tensor,
                 y_batch: torch.Tensor,
                 device: torch.device
    ):
        super().__init__()
        self._loss = partial(loss, X_batch=X_batch, y_batch=y_batch)
        self._variable_values = variable_values

        n = variable_values.flat_dim()
        self._shape = (n, n)
        self.device = device

    
    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian @ v using forward-over-reverse autodiff."""

        def grad_dot_v(var_vals: TensorDict) -> torch.Tensor:
            # Compute gradient of smooth_expr
            grad = torch.func.grad(lambda x: self._loss(x))(var_vals)
            return torch.dot(grad.to_flat_tensor(), v)

        # Differentiate grad_dot_v to get Hessian @ v
        hvp_td = torch.func.grad(grad_dot_v)(self._variable_values)
        return hvp_td.to_flat_tensor()
    
