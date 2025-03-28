from typing import Optional, Union

import scipy.sparse
import torch

from .ops import _get_row_slice, _csc_matmul


__all__ = ["SparseCSRTensor"]


class _SparseTensor:
    def __init__(
        self,
        data: torch.sparse.Tensor,
    ):
        self._data = data

    @property
    def data(self) -> torch.sparse.Tensor:
        return self._data

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    def __getitem__(self, indices: torch.Tensor) -> "_SparseTensor":
        if not isinstance(indices, torch.Tensor):
            raise IndexError("Row slice indices must be a tensor.")
        return _SparseTensor(_get_row_slice(self.data, indices))

    def __matmul__(self, v: torch.Tensor) -> torch.Tensor:
        if self.data.layout == torch.sparse_csr:
            return self.data @ v
        elif self.data.layout == torch.sparse_csc:
            return _csc_matmul(self.data, v)

    @property
    def T(self) -> "_SparseTensor":
        if self.data.layout == torch.sparse_csr:
            return self._get_csr_tranpose()
        elif self.data.layout == torch.sparse_csc:
            return self._get_csc_tranpose()
        else:
            raise ValueError("Unsupported sparse matrix layout.")

    def _get_csr_tranpose(self) -> "_SparseTensor":
        return _SparseTensor(
            torch.sparse_csc_tensor(
                self.data.crow_indices(),
                self.data.col_indices(),
                self.data.values(),
                size=(self.data.shape[1], self.data.shape[0]),
            )
        )

    def _get_csc_tranpose(self) -> "_SparseTensor":
        return _SparseTensor(
            torch.sparse_csr_tensor(
                self.data.ccol_indices(),
                self.data.row_indices(),
                self.data.values(),
                size=(self.data.shape[1], self.data.shape[0]),
            )
        )


class SparseCSRTensor(_SparseTensor):
    def __init__(
        self,
        data: Union[torch.sparse.Tensor, scipy.sparse.csr_matrix],
        device: Optional[Union[str, torch.device]] = None,
    ):
        # Check type of data
        if not self._is_sparse_csr_tensor(data) and not isinstance(
            data, scipy.sparse.csr_matrix
        ):
            raise TypeError(
                f"Unsupported data type {type(data)}. "
                "Expected torch.sparse_csr_tensor or scipy.sparse.csr_matrix."
            )

        # Check device
        if not isinstance(device, str) and not isinstance(device, torch.device):
            raise TypeError(
                f"Unsupported device type {type(device)}. Expected str or torch.device."
            )
        if self._is_sparse_csr_tensor(data):
            if device is not None and data.device != device:
                raise ValueError(
                    f"Device mismatch. Data is on device {data.device}, "
                    f"but device argument is {device}."
                )
            device = data.device
        else:
            if isinstance(device, str):
                device = torch.device(device)

        # Convert scipy sparse matrix to torch sparse tensor
        if isinstance(data, scipy.sparse.csr_matrix):
            data = self._from_scipy(data, device)
        super().__init__(data=data)

    def _is_sparse_csr_tensor(
        self, data: Union[torch.sparse.Tensor, scipy.sparse.csr_matrix]
    ) -> bool:
        if isinstance(data, torch.sparse.Tensor) and data.layout == torch.sparse_csr:
            return True
        return False

    def _from_scipy(
        self, data: scipy.sparse.csr_matrix, device: torch.device
    ) -> torch.sparse.Tensor:
        crow_indices = torch.tensor(data.indptr, dtype=torch.int64, device=device)
        col_indices = torch.tensor(data.indices, dtype=torch.int64, device=device)
        values = torch.tensor(data.data, device=device)
        data = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size=data.shape
        )
        return data
