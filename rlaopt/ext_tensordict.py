"""Extended TensorDict module."""

from __future__ import annotations

import tensordict as td_lib
import torch
from torch.utils._pytree import register_pytree_node


class TensorDict(td_lib.TensorDict):
    """Extended TensorDict with additional methods for optimization algorithms.

    This class extends the PyTorch tensordict.TensorDict with utilities
    commonly needed in optimization contexts, including:
    - Flat tensor operations (dot products, norms)
    - Conversions between TensorDict and flat tensor representations
    - Key structure manipulation

    All methods with 'flat__' suffix treat the TensorDict as a big vector,
    operating on all elements as if concatenated into a single 1D tensor.

    Examples:
        >>> td = TensorDict({
        ...     "a": torch.tensor([1.0, 2.0]),
        ...     "b": torch.tensor([3.0, 4.0])
        ... }, batch_size=[])
        >>>
        >>> # Flat operations
        >>> td.flat_dim()  # Total elements: 4
        >>> td.flat_norm()  # L2 norm: sqrt(1 + 4 + 9 + 16)
        >>>
        >>> # Vector conversion
        >>> flat_tensor = td.to_flat_tensor()  # tensor([1., 2., 3., 4.])
        >>> reconstructed = td.from_flat_tensor(flat_tensor)
    """

    def convert_target(self, target: TensorDict) -> TensorDict:
        """Relabel target to match this TensorDict's key structure."""
        return relabel_from_template(target, self)

    def flat_dim(self) -> int:
        """Get total number of elements across all tensors."""
        return sum(self[key].numel() for key in self.keys())

    def flat_dot(self, y: TensorDict) -> torch.Tensor:
        """Compute dot product treating both TensorDicts as flat vectors."""
        if self.keys() != y.keys():
            raise ValueError(
                "Cannot compute dot product!Keys of y do not agree with TensorDict"
            )
        return sum((self[key] * y[key]).sum() for key in self.keys())

    def flat_norm(self) -> torch.Tensor:
        """Compute L2 norm treating TensorDict as a flat 1D tensor."""
        # Get TensorDict of squared norms of each tensor.
        sq_norms_dict = self.norm() ** 2
        return torch.sqrt(sum(tensor_norm for tensor_norm in sq_norms_dict.values()))

    def to_flat_tensor(self) -> torch.Tensor:
        """Flatten all tensors into a single 1D tensor, that is a vector.

        Returns a 1D tensor containing all elements from all tensors
        in the TensorDict, concatenated in order.
        """
        values = list(self.values())
        if len(values) == 0:
            return torch.tensor([], device=self.device)
        return torch.cat([p.view(-1) for p in values])

    def from_flat_tensor(self, flat_tensor: torch.Tensor) -> TensorDict:
        """Reconstruct TensorDict from a flat 1D tensor, that is a vector.

        Unflattens the 1D tensor back into the original TensorDict structure,
        preserving shapes, keys, batch_size, and device.

        Args:
            flat_tensor: 1D tensor containing flattened values

        Returns:
            New TensorDict with the same structure as self
        """
        params_out = {}
        offset = 0
        for name, tensor in self.items():
            numel = tensor.numel()
            params_out[name] = flat_tensor[offset : offset + numel].view_as(tensor)
            offset += numel
        return TensorDict(params_out, batch_size=self.batch_size, device=self.device)


# PyTree registration for compatibility with torch.func
def _tensordict_flatten(td: TensorDict):
    """Flatten a TensorDict into (values, context) for pytree.

    Returns:
        values: list of tensors (the leaves)
        context: metadata needed to reconstruct (keys, batch_size, device)
    """
    keys = list(td.keys())
    values = [td[key] for key in keys]
    context = (keys, td.batch_size, td.device)
    return values, context


def _tensordict_unflatten(values, context):
    """Reconstruct a TensorDict from (values, context).

    Args:
        values: list of tensors
        context: (keys, batch_size, device) tuple
    """
    keys, batch_size, device = context
    td_dict = dict(zip(keys, values))
    return TensorDict(td_dict, batch_size=batch_size, device=device)


# Register the pytree node
register_pytree_node(TensorDict, _tensordict_flatten, _tensordict_unflatten)


def has_compatible_lengths(td1: TensorDict, td2: TensorDict) -> bool:
    """Check if two TensorDicts have the same number of leaves.

    Returns True if the number of key-value pairs in `td1` and `td2` match.
    """
    return len(list(td1.keys())) == len(list(td2.keys()))


def has_compatible_ordered_names(
    td1: TensorDict, td2: TensorDict, check_lengths: bool = True
) -> bool:
    """Check if two TensorDicts have the same keys in the same order.

    If `check_lengths` is True, verifies lengths before comparing keys.
    """
    if check_lengths and not has_compatible_lengths(td1, td2):
        print(
            f"Incompatible lengths: {len(td1)} vs {len(td2)}. "
            "Names cannot be compatible as td1 and td2 have different number of leaves."
        )
        return False

    false_indices = [
        (i, td1_key, td2_key)
        for i, (td1_key, td2_key) in enumerate(zip(td1.keys(), td2.keys()))
        if td1_key != td2_key
    ]

    if false_indices:
        print(
            f"Incompatible names at leaves (index, td1_key, td2_key): {false_indices}"
        )
        return False
    return True


def has_compatible_ordered_shapes(
    td1: TensorDict, td2: TensorDict, check_lengths: bool = True
) -> bool:
    """Check if two TensorDicts have the same tensor shapes in the same order.

    If `check_lengths` is True, verifies lengths before comparing shapes.
    """
    if check_lengths and not has_compatible_lengths(td1, td2):
        print(
            f"Incompatible lengths: {len(td1)} vs {len(td2)}. "
            "Shapes cannot be compatible as x and y have different number of leaves."
        )
        return False

    false_indices = [
        (i, td1_leaf.shape, td2_leaf.shape)
        for i, (td1_leaf, td2_leaf) in enumerate(zip(td1.values(), td2.values()))
        if td1_leaf.shape != td2_leaf.shape  # FIX: was td1.shape != td2.shape
    ]

    if false_indices:
        print(
            f"Incompatible shapes at leaves (index, td1_shape, td2_shape): "
            f"{false_indices}"
        )
        return False
    return True


def relabel_from_template(td: TensorDict, template: TensorDict) -> TensorDict:
    """Relabel the values in `td` using the key structure from `template`.

    This is useful when you want to preserve the values from `td`
    but match the key names (in order) from `template`.

    Requirements:
    - `td` and `template` must have the same number of elements.
    - Corresponding tensors must have the same shapes (in order).

    Returns a new TensorDict with keys from `template` and values from `td`.

    Raises:
        ValueError: if shapes or lengths are incompatible.
    """
    if has_compatible_ordered_shapes(template, td):
        return TensorDict(
            dict(zip(template.keys(), td.values())),
            batch_size=template.batch_size,
        )
    else:
        raise ValueError("Incompatible ordered shapes between x and template.")
