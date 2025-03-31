from typing import Optional, Union

# import scipy.sparse as sp
from scipy.sparse import csr_matrix, csr_array, csc_array
import torch

from .ops import _get_row_slice, _csc_matmul


SP_NAME = "scipy.sparse"


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
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    def to(self, *args, **kwargs) -> "_SparseTensor":
        return _SparseTensor(self.data.to(*args, **kwargs))

    def cpu(self, *args, **kwargs) -> "_SparseTensor":
        return _SparseTensor(self.data.cpu(*args, **kwargs))

    def cuda(self, *args, **kwargs) -> "_SparseTensor":
        return _SparseTensor(self.data.cuda(*args, **kwargs))

    def scipy(self, *args, **kwargs) -> Union[csr_array, csc_array]:
        """Convert to scipy sparse array."""
        if self.data.layout == torch.sparse_csr:
            return self._to_scipy_csr(*args, **kwargs)
        elif self.data.layout == torch.sparse_csc:
            ccol_indices = self.data.ccol_indices().numpy(*args, **kwargs)
            row_indices = self.data.row_indices().numpy(*args, **kwargs)
            values = self.data.values().numpy(*args, **kwargs)
            return csc_array((values, row_indices, ccol_indices), shape=self.data.shape)
        else:
            raise ValueError("Unsupported sparse matrix layout.")

    def _to_scipy_csr(self, *args, **kwargs) -> csr_array:
        crow_indices = self.data.crow_indices().numpy(*args, **kwargs)
        col_indices = self.data.col_indices().numpy(*args, **kwargs)
        values = self.data.values().numpy(*args, **kwargs)
        return csr_array((values, col_indices, crow_indices), shape=self.data.shape)

    def _to_scipy_csc(self, *args, **kwargs) -> csc_array:
        ccol_indices = self.data.ccol_indices().numpy(*args, **kwargs)
        row_indices = self.data.row_indices().numpy(*args, **kwargs)
        values = self.data.values().numpy(*args, **kwargs)
        return csc_array((values, row_indices, ccol_indices), shape=self.data.shape)

    def __getitem__(self, indices: torch.Tensor) -> "_SparseTensor":
        # print(f"Indices: {indices}")
        # print(f"Indices type: {type(indices)}")
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
        data: Union[torch.sparse.Tensor, csr_matrix, csr_array],
        device: Optional[Union[str, torch.device]] = None,
    ):
        # Check type of data
        if not self._is_sparse_csr_tensor(data) and not isinstance(
            data, (csr_matrix, csr_array)
        ):
            raise TypeError(
                f"Unsupported data type {type(data).__name__}. "
                f"Expected torch.sparse_csr_tensor, {SP_NAME}.csr_matrix, "
                f"or {SP_NAME}.csr_array."
            )

        # Check device
        if not isinstance(device, str) and not isinstance(device, torch.device):
            raise TypeError(
                f"Unsupported device type {type(device).__name__}. "
                "Expected str or torch.device."
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
        if isinstance(data, (csr_matrix, csr_array)):
            data = self._from_scipy(data, device)
        super().__init__(data=data)

    def _is_sparse_csr_tensor(
        self, data: Union[torch.sparse.Tensor, csr_matrix, csr_array]
    ) -> bool:
        if isinstance(data, torch.sparse.Tensor) and data.layout == torch.sparse_csr:
            return True
        return False

    def _from_scipy(
        self, data: Union[csr_matrix, csr_array], device: torch.device
    ) -> torch.sparse.Tensor:
        crow_indices = torch.tensor(data.indptr, dtype=torch.int64, device=device)
        col_indices = torch.tensor(data.indices, dtype=torch.int64, device=device)
        values = torch.tensor(data.data, device=device)
        data = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size=data.shape
        )
        return data
