from typing import Optional, Tuple

import torch

from . import prox_grad_func
from ..configs import ProxGradConfig
from ...expression.expression import Expression, AddExpression
from ...params.params import Params

class ProximalGradient:
    def __init__(
            self, 
            obj: Expression|AddExpression, 
            config: ProxGradConfig
    ):
        self.config = config
        self._step = prox_grad_func._build_step(
            obj, 
            config.use_acceleration, 
            config.use_linesearch
        )
    
    def init_state(self, params: Params):
        return prox_grad_func._init_state(
            params, 
            self.config.eta, 
            self.config.use_acceleration
        )
    
    def step(self, params: Params, state: prox_grad_func.ProxGradState
    )->Tuple[Params, prox_grad_func.ProxGradState]:
        return self._step(params, state)
    
    def solve(
            self, 
            obj: Expression|AddExpression, 
            init_params: Optional[Params]=None
    )-> Tuple[Params, torch.Tensor]:
        return prox_grad_func.proximal_gradient(obj, self.config, init_params)