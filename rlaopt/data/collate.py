import torch
from torch.utils.data._utils.collate import default_collate


class IndexTrackingCollate:
    """Collate function to stack (data, target, index) tuples into batch tensors."""

    def __call__(self, batch):
        """Args:
            batch (list): A list of tuples, where each tuple is the output of
                          Dataset.__getitem__: (X_i, y_i, index_i)

        Returns:
            tuple: (X_batch, y_batch, index_batch)
        """
        # 1. Separate the indices from the data/target tuples
        # The first two elements are data (X, y), the last is the index.
        data_target_only = [(item[0], item[1]) for item in batch]
        index_list = [item[2] for item in batch]

        # 2. Use PyTorch's default collate function to handle X and y
        # This handles tensors, numpy arrays, and nested structures correctly.
        data_target_batch = default_collate(data_target_only)

        # 3. Stack the indices into a LongTensor
        index_batch = torch.stack(index_list)

        # 4. Combine and return: (X_batch, y_batch, index_batch)
        return *data_target_batch, index_batch
