from typing import Dict, NamedTuple

import torch

TensorDict = Dict[str, torch.Tensor]
OptimState = NamedTuple