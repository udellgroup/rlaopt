from functools import partial

import torch

from .datasets import Dataset, BatchedDataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Dataset | BatchedDataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
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
        if not isinstance(dataset, Dataset) and not isinstance(dataset, BatchedDataset):
            raise TypeError(
                f"Dataset must be of type Dataset or BatchedDataset but recieved {type(dataset)}"
            )

        if isinstance(dataset, Dataset):
            self._y = partial(get_training_labels, loader=self, in_memory=True)
        else:
            self._y = partial(get_training_labels, loader=self, in_memory=False)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
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

    @property
    def y(self):
        return self._y()


def get_training_labels(loader: "DataLoader", in_memory: bool):
    if in_memory:
        return loader.dataset.y
    else:
        training_labels = []
        for _, y_batch in loader:
            training_labels.append(y_batch)
        return torch.cat(training_labels, dim=0)
