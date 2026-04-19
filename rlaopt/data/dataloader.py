"""Custom DataLoader implementation with extended functionality for Dataset types.

This module provides a DataLoader class that extends PyTorch's standard DataLoader
to work with custom Dataset and BatchedDataset types, offering efficient access to
training labels based on the dataset type.
"""

from functools import partial

import torch

from rlaopt.data.datasets import BatchedDataset, Dataset

from .collate import IndexTrackingCollate


class DataLoader(torch.utils.data.DataLoader):
    """Extended PyTorch DataLoader with custom Dataset support and lazy labels.

    This DataLoader extends torch.utils.data.DataLoader to work specifically
    with Dataset and BatchedDataset types, providing additional functionality
    for accessing training labels efficiently based on the dataset type.

    Args:
        dataset: A Dataset or BatchedDataset instance to load data from.
        batch_size: Number of samples per batch. Default: 1.
        shuffle: Whether to shuffle the data at every epoch. Default: None.
        sampler: Strategy to draw samples from the dataset. Default: None.
        batch_sampler: Strategy to draw batches of samples. Default: None.
        num_workers: Number of subprocesses for data loading. Default: 0.
        pin_memory: Whether to copy tensors into CUDA pinned memory. Default: False.
        drop_last: Whether to drop the last incomplete batch. Default: False.
        timeout: Timeout value for collecting a batch from workers. Default: 0.
        worker_init_fn: Function called on each worker subprocess. Default: None.
        multiprocessing_context: Multiprocessing context for workers. Default: None.
        generator: Random number generator for sampling. Default: None.
        prefetch_factor: Number of batches loaded in advance by each worker.
            Default: None.
        persistent_workers: Whether to keep workers alive between epochs.
            Default: False.
        pin_memory_device: Device where tensors should be pinned. Default: "".
        in_order: Whether to maintain order when loading data. Default: True.

    Raises:
        TypeError: If dataset is not an instance of Dataset or BatchedDataset.

    Attributes:
        y: Property that returns all training labels from the dataset. For
           Dataset instances, labels are retrieved directly from memory. For
           BatchedDataset instances, labels are collected by iterating through
           all batches.

    Example:
        >>> dataset = MyDataset(...)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> labels = loader.y  # Access all training labels
        >>> for batch_x, batch_y in loader:
        ...     # Training loop
    """

    def __init__(
        self,
        dataset: Dataset | BatchedDataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
        in_order=True,
    ):
        """Initialize the DataLoader with the given dataset and parameters."""
        if not isinstance(dataset, Dataset) and not isinstance(dataset, BatchedDataset):
            raise TypeError(
                f"Dataset must be of type Dataset or BatchedDataset but "
                f"received {type(dataset).__name__}"
            )

        # Inject the custom collate function so batches carry sample indices.
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            IndexTrackingCollate(),
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            in_order=in_order,
        )

        # 3. Setup label access (_y property)
        if isinstance(dataset, Dataset):
            # Assumes _get_training_labels is defined and handles the logic
            # for in-memory data
            self._y = partial(_get_training_labels, loader=self, in_memory=True)
        else:
            # Assumes _get_training_labels is defined and handles the logic
            # for iterating batched data
            self._y = partial(_get_training_labels, loader=self, in_memory=False)

        self.data_iter = iter(self)
        self._shuffle = shuffle

    def get_batch(self) -> tuple[torch.Tensor, ...]:
        """Fetch the next batch from the DataLoader.

        Automatically resets the iterator upon consumption (end of epoch).
        """
        try:
            # Attempt to get the next batch from the current iterator
            return next(self.data_iter)
        except StopIteration:
            # Iterator exhausted - reset to beginning for next epoch
            self.data_iter = iter(self)
            return next(self.data_iter)

    @property
    def y(self):
        """Get all training labels from the dataset."""
        return self._y()


def _get_training_labels(loader: DataLoader, in_memory: bool):
    """Retrieve all training labels from a DataLoader.

    Extracts training labels from the DataLoader's dataset using the appropriate
    method based on whether the dataset supports in-memory label access.

    Args:
        loader: The DataLoader instance to extract labels from.
        in_memory: If True, retrieves labels directly from dataset.y (for Dataset).
                   If False, iterates through all batches to collect labels
                   (for BatchedDataset).

    Returns:
        torch.Tensor: A tensor containing all training labels from the dataset.
                      For batched collection, labels are concatenated along dimension 0.

    Note:
        This function is typically called internally via the DataLoader.y property
        rather than being invoked directly.
    """
    if in_memory:
        return loader.dataset.y
    else:
        training_labels = []
        for _, y_batch, _ in loader:
            training_labels.append(y_batch)
        return torch.cat(training_labels, dim=0)
