import torch


def check_and_move_to_device(
    tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if tensor.device != device: tensor = tensor.to(device)
    return tensor


def move_to_source_device(
    tensors: torch.Tensor | tuple[torch.Tensor], source_tensor: torch.Tensor
) -> torch.Tensor | tuple[torch.Tensor]:
    """Move tensors to the same device as the source tensor."""
    target_device = source_tensor.device
    if isinstance(tensors, torch.Tensor):
        return check_and_move_to_device(tensors, target_device)
    else:
        tensors = tuple(check_and_move_to_device(t, target_device) for t in tensors)
        return tensors
