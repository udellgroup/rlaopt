from typing import List, Optional, Union

from scipy.sparse import csr_matrix, csr_array, csc_array
import torch

from .ops import _get_row_slice, _csc_matmul
from .utils import _convert_indices_to_tensor

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

    def _is_csr(self) -> bool:
        return self.data.layout == torch.sparse_csr

    def _is_csc(self) -> bool:
        return self.data.layout == torch.sparse_csc

    def _to_scipy_csr(self, *args, **kwargs) -> csr_array:
        crow_indices = self.data.crow_indices().cpu().numpy(*args, **kwargs)
        col_indices = self.data.col_indices().cpu().numpy(*args, **kwargs)
        values = self.data.values().cpu().numpy(*args, **kwargs)
        return csr_array((values, col_indices, crow_indices), shape=self.data.shape)

    def _to_scipy_csc(self, *args, **kwargs) -> csc_array:
        ccol_indices = self.data.ccol_indices().cpu().numpy(*args, **kwargs)
        row_indices = self.data.row_indices().cpu().numpy(*args, **kwargs)
        values = self.data.values().cpu().numpy(*args, **kwargs)
        return csc_array((values, row_indices, ccol_indices), shape=self.data.shape)

    def scipy(self, *args, **kwargs) -> Union[csr_array, csc_array]:
        """Convert to scipy sparse array.

        Args:
            *args: Additional arguments to pass to torch.Tensor.numpy()
            **kwargs: Additional keyword arguments to pass to torch.Tensor.numpy()

        Returns:
            Union[csr_array, csc_array]: The converted scipy sparse array

        Raises:
            ValueError: If the sparse tensor is not in CSR or CSC format
        """
        if self._is_csr():
            return self._to_scipy_csr(*args, **kwargs)
        elif self._is_csc():
            return self._to_scipy_csc(*args, **kwargs)
        else:
            raise ValueError("Unsupported sparse matrix layout.")

    def __getitem__(
        self, indices: Union[torch.Tensor, slice, int, List[int]]
    ) -> "_SparseTensor":
        """Get rows from a sparse tensor using indices.

        Args:
            indices: Can be torch.Tensor, slice, int, or list of indices

        Returns:
            _SparseTensor: A new sparse tensor with the selected rows

        Raises:
            ValueError: If tensor is not in CSR format
            TypeError: If indices format is invalid
            IndexError: If indices are out of bounds
        """
        if not self._is_csr():
            raise ValueError("Row slicing is only supported for CSR layout.")

        # Convert indices to tensor using the utility function
        tensor_indices = _convert_indices_to_tensor(
            indices, num_rows=self.data.shape[0]
        )

        return _SparseTensor(_get_row_slice(self.data, tensor_indices.to(self.device)))

    def _check_matmul_input_shape(self, v: torch.Tensor) -> None:
        if v.ndim not in [1, 2]:
            raise ValueError(f"v must be a 1D or 2D tensor. Received {v.ndim}D tensor.")

    def __matmul__(self, v: torch.Tensor) -> torch.Tensor:
        self._check_matmul_input_shape(v)
        if self._is_csr():
            return self.data @ v
        elif self._is_csc():
            return _csc_matmul(self.data, v)

    def __rmatmul__(self, v: torch.Tensor) -> torch.Tensor:
        self._check_matmul_input_shape(v)
        if v.ndim == 1:
            return self.T @ v
        elif v.ndim == 2:
            return (self.T @ v.T).T

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

    @property
    def T(self) -> "_SparseTensor":
        """Transpose of the sparse tensor.

        Returns:
            _SparseTensor: Transposed sparse tensor

        Raises:
            ValueError: If the sparse tensor is not in CSR or CSC format
        """
        if self._is_csr():
            return self._get_csr_tranpose()
        elif self._is_csc():
            return self._get_csc_tranpose()
        else:
            raise ValueError("Unsupported sparse matrix layout.")


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
