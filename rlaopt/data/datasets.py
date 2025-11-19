"""Dataset classes for handling various data modalities in machine learning workflows.

This module provides flexible dataset classes that bridge the gap between common data
formats (numpy, pandas) and PyTorch tensors, with support for both in-memory and
out-of-memory datasets.

Classes:
    AbstractDataset: Base abstract class defining the dataset interface.
    Dataset: In-memory dataset supporting numpy, pandas, and torch tensors.
    BatchedDataset: Abstract class for datasets too large to fit in memory.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing_extensions import Self


class BaseDataset(torch.utils.data.Dataset, ABC):
    """Abstract base class for all dataset types.

    Defines the common interface that all dataset classes must implement,
    including properties for introspecting dataset dimensions.
    """

    @property
    @abstractmethod
    def num_samples(self):
        """int: Total number of samples in the dataset."""
        pass

    @property
    @abstractmethod
    def feature_dimension(self):
        """Int or tuple: Dimension(s) of the feature space."""
        pass

    @property
    @abstractmethod
    def target_dimension(self):
        """Int or tuple: Dimension(s) of the target space."""
        pass


class BatchedDataset(BaseDataset, ABC):
    """Abstract base class for datasets that are too large to fit in memory.

    Subclasses must implement __getitem__ and __len__ following torch.utils.data.Dataset
    conventions, as well as properties to introspect feature and target dimensions.

    This class is designed for datasets that can only be accessed in batches,
    where loading the entire dataset into memory is infeasible.

    Examples:
        >>> class MyLargeDataset(BatchedDataset):
        ...     def __init__(self, data_path):
        ...         self.data_path = data_path
        ...         # Load metadata to determine shapes
        ...
        ...     def __getitem__(self, idx):
        ...         # Load sample(s) from disk
        ...         return X, y
        ...
        ...     def __len__(self):
        ...         return self.total_samples
        ...
        ...     @property
        ...     def feature_dimension(self):
        ...         return self.n_features
        ...
        ...     @property
        ...     def target_dimension(self):
        ...         return self.n_targets
    """

    def __init__(self):
        """Initialize BatchedDataset."""
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx):
        """Retrieve a sample or batch of samples.

        Args:
            idx (int or slice): Index or slice of samples to retrieve.

        Returns:
            tuple: (X, y) where X is features and y is target(s).
        """
        pass

    @abstractmethod
    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        pass

    @property
    def num_samples(self):
        """int: Total number of samples in the dataset."""
        return len(self)

    @property
    @abstractmethod
    def feature_dimension(self):
        """Int or tuple: Dimension(s) of the feature space.

        Subclasses should implement this by inspecting metadata or a sample.
        """
        pass

    @property
    @abstractmethod
    def target_dimension(self):
        """Int or tuple: Dimension(s) of the target space.

        Subclasses should implement this by inspecting metadata or a sample.
        """
        pass


class Dataset(BaseDataset, torch.utils.data.TensorDataset):
    """In-memory dataset for classical machine learning tasks.

    Handles data matrices with labels/response vectors that fit in memory.
    Automatically converts numpy arrays and pandas DataFrames/Series to PyTorch tensors.
    Suitable for GLMs, classical statistical problems, and convex optimization tasks.

    Args:
        X (Tensor, np.ndarray, pd.DataFrame, or pd.Series): Feature matrix of shape
            (n_samples, n_features).
        y (Tensor, np.ndarray, pd.DataFrame, or pd.Series): Target array of shape
            (n_samples, ...). Can be any dimensionality.
        device (str or torch.device, optional): Device to place tensors on
            (e.g., 'cpu', 'cuda', 'cuda:0'). Defaults to None.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.

    Raises:
        ValueError: If X is not 2-dimensional or if X and y have mismatched sample sizes.

    Examples:
        >>> # From numpy
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randn(100)
        >>> data = Dataset(X, y)

        >>> # From pandas with device specification
        >>> df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4]})
        >>> y = pd.Series([5, 6])
        >>> data = Dataset(df, y, device='cuda')

        >>> # Multi-target
        >>> y_multi = np.random.randn(100, 3)
        >>> data = Dataset(X, y_multi)
    """

    def __init__(
        self,
        X: Tensor | np.ndarray | pd.DataFrame | pd.Series,
        y: Tensor | np.ndarray | pd.DataFrame | pd.Series,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize Dataset with feature matrix and target array."""
        # Convert pandas to numpy first
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        # Convert X to tensor with specified dtype
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(dtype)
        elif isinstance(X, Tensor) and X.dtype != dtype:
            X = X.to(dtype)

        # Carefully handle dtype of y
        # Preserve integer types for classification, use specified dtype for regression
        if isinstance(y, np.ndarray):
            # Check if y contains integer data (for classification)
            if np.issubdtype(y.dtype, np.integer):
                y = torch.from_numpy(y).long()
            else:
                y = torch.from_numpy(y).to(dtype)
        elif isinstance(y, Tensor):
            # Preserve long/int dtypes, convert others to specified dtype
            if y.dtype in [torch.int32, torch.int64, torch.long]:
                y = y.long()
            elif y.dtype != dtype:
                y = y.to(dtype)

        # Handle device placement
        if device is not None:
            device = torch.device(device)
            X = X.to(device)
            y = y.to(device)

        # Validate X dimensions
        if X.ndim != 2:
            raise ValueError("Input data tensor X must be 2-dimensional.")

        # Validate matching sample sizes
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        super().__init__(X, y)

    @classmethod
    def from_numpy(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        device: str | torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Create a Dataset from numpy arrays.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target array of shape (n_samples, ...). Can be any
                dimensionality.
            device (str or torch.device, optional): Device to place tensors on
                (e.g., 'cpu', 'cuda', 'cuda:0'). Defaults to None.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to
                torch.float32.

        Returns:
            Dataset: Dataset instance with data on specified device.

        Examples:
            >>> X = np.random.randn(100, 10)
            >>> y = np.random.randn(100)
            >>> data = Dataset.from_numpy(X, y, device='cuda')
        """
        return cls(X, y, device=device, dtype=dtype)

    @classmethod
    def from_pandas(
        cls,
        X: pd.DataFrame | pd.Series,
        y: pd.DataFrame | pd.Series,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Create a Dataset from pandas DataFrames or Series.

        Args:
            X (pd.DataFrame or pd.Series): Feature data of shape
                (n_samples, n_features).
            y (pd.DataFrame or pd.Series): Target data of shape (n_samples, ...).
                Can be any dimensionality.
            device (str or torch.device, optional): Device to place tensors on
                (e.g., 'cpu', 'cuda', 'cuda:0'). Defaults to None.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to
                torch.float32.

        Returns:
            Dataset: Dataset instance with data on specified device.

        Examples:
            >>> # From separate DataFrames
            >>> df_X = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
            >>> df_y = pd.Series([7, 8, 9])
            >>> data = Dataset.from_pandas(df_X, df_y)

            >>> # From a single DataFrame using column selection
            >>> df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'y': [7, 8, 9]})
            >>> data = Dataset.from_pandas(df[['x1', 'x2']], df['y'])

            >>> # Multi-target
            >>> df_multi = pd.DataFrame({'x1': [1, 2], 'y1': [3, 4], 'y2': [5, 6]})
            >>> data = Dataset.from_pandas(df_multi[['x1']], df_multi[['y1', 'y2']])
        """
        return cls(X, y, device=device, dtype=dtype)

    def to(self, device: str | torch.device) -> Self:
        """Move dataset to specified device.

        Args:
            device (str or torch.device): Target device (e.g., 'cpu', 'cuda',
                'cuda:0').

        Returns:
            Dataset: New Dataset instance on the specified device.

        Examples:
            >>> data = Dataset(X, y, device='cpu')
            >>> data_gpu = data.to('cuda')
            >>> print(data_gpu.device)  # cuda:0
        """
        device = torch.device(device)
        X_new = self.X.to(device)
        y_new = self.y.to(device)
        return Dataset(X_new, y_new)

    @property
    def device(self) -> torch.device:
        """torch.device: Device where the dataset tensors are stored."""
        return self.X.device

    @property
    def dtype(self) -> torch.dtype:
        """torch.dtype: Data type of the dataset tensors."""
        return self.X.dtype

    @property
    def num_samples(self):
        """int: Total number of samples in the dataset."""
        return self.tensors[0].shape[0]

    @property
    def feature_dimension(self):
        """int: Number of features in the dataset."""
        return self.tensors[0].shape[1]

    @property
    def target_dimension(self):
        """Int or tuple: Dimension(s) of the target.

        Returns 1 for 1D targets (shape (n,)). For multi-dimensional targets,
        returns a tuple of dimensions excluding the sample dimension.

        Examples:
            >>> # 1D target
            >>> y = torch.randn(100)
            >>> data = Dataset(X, y)
            >>> data.target_dimension  # 1

            >>> # 2D multi-target
            >>> y = torch.randn(100, 5)
            >>> data = Dataset(X, y)
            >>> data.target_dimension  # (5,)

            >>> # 3D target (e.g., images)
            >>> y = torch.randn(100, 3, 28, 28)
            >>> data = Dataset(X, y)
            >>> data.target_dimension  # (3, 28, 28)
        """
        if self.tensors[1].ndim == 1:
            return 1
        return self.tensors[1].shape[1:]  # Return all dimensions after batch

    @property
    def X(self):
        """Tensor: Feature matrix of shape (n_samples, n_features)."""
        return self.tensors[0]

    @property
    def y(self):
        """Tensor: Target array of shape (n_samples, ...)."""
        return self.tensors[1]

    def __repr__(self):
        """Return string representation of the dataset.

        Returns:
            str: String showing dataset dimensions and device.
        """
        return (
            f"{self.__class__.__name__}("
            f"num_samples={self.num_samples}, "
            f"feature_dimension={self.feature_dimension}, "
            f"target_dimension={self.target_dimension}, "
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )
