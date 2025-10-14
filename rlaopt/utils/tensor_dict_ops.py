import torch

from rlaopt._typing import TensorDict


def add(x: TensorDict, y: TensorDict) -> TensorDict:
    return {name: x[name] + y[name] for name in x}


def sub(x: TensorDict, y: TensorDict) -> TensorDict:
    return {name: x[name] - y[name] for name in x}


def elem_mul(x: TensorDict, y: TensorDict) -> TensorDict:
    return {name: x[name] * y[name] for name in x}


def elem_div(x: TensorDict, y: TensorDict) -> TensorDict:
    return {name: x[name] / y[name] for name in x}


def scal_mul(x: TensorDict, a: float) -> TensorDict:
    return {name: a * x_leaf for name, x_leaf in x.items()}


def dot(x: TensorDict, y: TensorDict):
    return torch.sum(torch.stack([torch.sum(x[name] * y[name]) for name in x.keys()]))


def elem_norm(x: TensorDict) -> torch.Tensor:
    return torch.sqrt(dot(x, x))


def clone(x: TensorDict) -> TensorDict:
    {name: x_leaf.clone() for name, x_leaf in x.items()}


def shapes(x: TensorDict) -> dict[str, torch.Size]:
    return {name: p.shape for name, p in x.items()}


def has_compatible_lengths(x: TensorDict, y: TensorDict) -> bool:
    """
    Check if two TensorDicts have the same number of leaves.

    Returns True if the number of key-value pairs in `x` and `y` match.
    """
    return len(x) == len(y)


def has_compatible_ordered_names(
    x: TensorDict, y: TensorDict, check_lengths: bool = True
) -> bool:
    """
    Check if two TensorDicts have the same keys in the same order.

    If `check_lengths` is True, verifies lengths before comparing keys.
    """
    if check_lengths and not has_compatible_lengths(x, y):
        print(
            f"Incompatible lengths: {len(x)} vs {len(y)}. "
            "Names cannot be compatible as x and y have different number of leaves."
        )
        return False

    false_indices = [
        (i, x_key, y_key)
        for i, (x_key, y_key) in enumerate(zip(x.keys(), y.keys()))
        if x_key != y_key
    ]

    if false_indices:
        print(f"Incompatible names at leaves (index, x_key, y_key): {false_indices}")
        return False
    return True


def has_compatible_ordered_shapes(
    x: TensorDict, y: TensorDict, check_lengths: bool = True
) -> bool:
    """
    Check if two TensorDicts have the same tensor shapes in the same order.

    If `check_lengths` is True, verifies lengths before comparing shapes.
    """
    if check_lengths and not has_compatible_lengths(x, y):
        print(
            f"Incompatible lengths: {len(x)} vs {len(y)}. "
            "Shapes cannot be compatible as x and y have different number of leaves."
        )
        return False

    false_indices = [
        (i, x_leaf.shape, y_leaf.shape)
        for i, (x_leaf, y_leaf) in enumerate(zip(x.values(), y.values()))
        if x_leaf.shape != y_leaf.shape
    ]

    if false_indices:
        print(
            f"Incompatible shapes at leaves (index, x_shape, y_shape): {false_indices}"
        )
        return False
    return True


def is_compatible(x: TensorDict, y: TensorDict) -> bool:
    """
    Check full compatibility between two TensorDicts.

    Returns True if:
    - They have the same number of keys,
    - Keys match in order,
    - Tensor shapes match in order.

    Otherwise returns False, with diagnostics printed from the sub-checks.
    """
    return has_compatible_ordered_names(x, y) and has_compatible_ordered_shapes(
        x, y, check_lengths=False
    )


def relabel_from_template(x: TensorDict, template: TensorDict) -> TensorDict:
    """
    Relabel the values in `x` using the key structure from `template`.

    This is useful when you want to preserve the values from `x`
    but match the key names (in order) from `template`.

    Requirements:
    - `x` and `template` must have the same number of elements.
    - Corresponding tensors must have the same shapes (in order).

    Returns a new TensorDict with keys from `template` and values from `x`.

    Raises:
        ValueError: if shapes or lengths are incompatible.
    """
    if has_compatible_ordered_shapes(template, x):
        return {name: x_leaf for name, x_leaf in zip(template.keys(), x.values())}
    else:
        raise ValueError("Incompatible ordered shapes between x and template.")


def dim(x: TensorDict) -> int:
    return sum(p.numel() for p in x.values())


def equal(x: TensorDict, y: TensorDict, tol: float = 1e-6) -> bool:
    return all(torch.allclose(x[name], y[name], atol=tol) for name in x)


def zero_like(params: TensorDict) -> TensorDict:
    return {name: torch.zeros_like(t) for name, t in params.items()}


def flatten(params: TensorDict) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params.values()])


def unflatten(vec: torch.Tensor, template: TensorDict) -> TensorDict:
    params_out = {}
    offset = 0
    for name, tensor in template.items():
        numel = tensor.numel()
        params_out[name] = vec[offset : offset + numel].view_as(tensor)
        offset += numel
    return params_out


def dict_map(x: TensorDict, f, *args, **kwargs) -> TensorDict:
    return {name: f(p, *args, **kwargs) for name, p in x.items()}
