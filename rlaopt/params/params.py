from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch.utils._pytree import register_pytree_node

from . import param_utils

class Params:
    def __init__(self, params: Dict[str, torch.Tensor]):
        self._params = params
        self._dim = param_utils.params_dim(params)
        self._names = params.keys()
        self._data = params.values()
        self._shapes = [leaf.shape for leaf in self._data]
    
    def clone(self) -> "Params":
        return Params(param_utils.clone(self._params))
    
    def flatten(self) -> torch.Tensor:
        return param_utils.flatten(self._params)

    def unflatten(self, v: torch.Tensor) -> "Params":
        return Params(param_utils.unflatten(v, self._params))
    
    def to(self, device) -> "Params":
        return Params({name: leaf.to(device) for name, leaf in self._params.items()})

    def detach(self) -> "Params":
        return Params({name: leaf.detach() for name, leaf in self._params.items()})
    
    def is_compatible(self, y: "Params") -> bool:
        if self.names != y.names:
            return False
        elif not self.has_compatible_shapes(y):
            return False
        else:
            return True
            
    def has_compatible_shapes(self, y: "Params") -> bool:
        if self.dim != y.dim:
            return False
        elif self.shapes != y.shapes:
            return False
        else:
            return True
    
    def fmap(self, f: Callable[[torch.Tensor], torch.Tensor]):
        return Params(param_utils.params_map(self._params, f))

    def dot(self, y: "Params")-> torch.Tensor:
        return param_utils.dot(self._params, y._params)
    
    def norm(self):
        return  param_utils.elem_norm(self._params)
    
    @property 
    def data(self):
        return self._data

    @property
    def dim(self):
        return self._dim
    
    @property
    def names(self):
        return self._names
    
    @property
    def shapes(self):
        return self._shapes
    
    @property
    def value(self):
        return self._params
    
    def __add__(self, y: "Params")-> "Params":
        return Params(param_utils.add(self._params, y._params))
    
    def __sub__(self, y: "Params")-> "Params":
        return Params(param_utils.sub(self._params, y._params))
    
    def __mul__(self, y: Union["Params", float]) -> "Params":
        if isinstance(y, Params):
           return Params(param_utils.elem_mul(self._params, y._params))
        else:
            return Params(param_utils.scal_mul(self._params, y))
    
    def __rmul__(self, y: Union["Params", float]) -> "Params":
        return self * y
    
    def __div__(self, y: Union["Params", float]) -> "Params":
        if isinstance(y, Params):
            return Params(param_utils.elem_div(self._params, y._params))
        else:
            return Params(param_utils.scal_mul(self._params, 1 / y))
    
    def __neg__(self):
        return Params(param_utils.scal_mul(self._params, -1.0))
    
    def __eq__(self, y: "Params"):
        return param_utils.params_equal(self._params, y._params)


def flatten_fn(p: Params) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    flat_tensors = list(p.value.values())  # List of tensors as individual leaves
    context = {"keys": list(p.value.keys())}
    return flat_tensors, context

def unflatten_fn(tensors: List[torch.Tensor], context: Dict[str, Any]) -> Params:
    keys = context["keys"]
    param_dict = {k: t for k, t in zip(keys, tensors)}
    return Params(param_dict)

register_pytree_node(Params, flatten_fn, unflatten_fn)