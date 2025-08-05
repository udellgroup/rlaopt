from typing import Callable, Dict, Iterable

import torch

Parameters = Dict[str, torch.Tensor]

def axpy(a: torch.Tensor, x: Iterable[torch.Tensor], y: Parameters) -> Parameters:
    return {name: y_leaf + a * x_leaf for (name, y_leaf), x_leaf in zip(y.items(), x)}

def add(x: Parameters, y: Parameters) -> Parameters:
    return {name: x[name] + y[name] for name in x}

def sub(x: Parameters, y: Parameters) -> Parameters:
    return {name: x[name] - y[name] for name in x}

def elem_mul(x: Parameters, y: Parameters) -> Parameters:
    return {name: x[name] * y[name] for name in x}

def elem_div(x: Parameters, y: Parameters) -> Parameters:
    return {name: x[name] / y[name] for name in x}

def scal_mul(x: Parameters, a: float) -> Parameters:
    return {name: a * x_leaf for name, x_leaf in x.items()}

def clone(x: Parameters) -> Parameters:
    {name: x_leaf.clone() for name, x_leaf in x.items()}
    
def params_dim(x: Parameters) -> int:
    return sum(p.numel() for p in x.values())

def params_equal(x: Parameters, y: Parameters, tol: float = 1e-6) -> bool:
    return all(torch.allclose(x[name], y[name], atol=tol) for name in x)

def params_map(x: Parameters, f, *args, **kwargs) -> Parameters:
    return {name: f(p, *args, **kwargs) for name, p in x.items()}

def dot(x: Parameters, y: Parameters):
    xy = [x[name] * y[name] for name in x]
    return torch.sum(torch.stack(xy))

def elem_norm(x: Parameters) -> torch.Tensor:
    return torch.sqrt(dot(x,x))

def zero_like(params: Parameters) -> Parameters:
    return {name: torch.zeros_like(t) for name, t in params.items()}

def flatten(params: Parameters) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params.values()])

def unflatten(vec: torch.Tensor, template: Parameters) -> Parameters:
    params_out = {}
    offset = 0
    for name, tensor in template.items():
        numel = tensor.numel()
        params_out[name] = vec[offset:offset+numel].view_as(tensor)
        offset += numel
    return params_out
 