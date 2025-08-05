from dataclasses import dataclass

import torch

@dataclass(kw_only=True,frozen=False)
class ProxGradConfig:
    eta: torch.Tensor = None
    max_iters: int = 5000
    tol: float = 1e-4
    use_acceleration: bool = False
    use_linesearch: bool = True 
