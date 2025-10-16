from typing import NamedTuple

import torch

TensorDict = dict[str, torch.Tensor]
OptimState = NamedTuple
LinSysState = NamedTuple
