"""Extended TensorDict module."""

import tensordict as td_lib
import torch
from torch.utils._pytree import register_pytree_node
from typing_extensions import Self


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

    def flat_dim(self) -> int:
        """Get total number of elements across all tensors."""
        return sum(self[key].numel() for key in self.keys())

    def flat_dot(self, y: Self) -> torch.Tensor:
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

    def from_flat_tensor(self, flat_tensor: torch.Tensor) -> Self:
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
