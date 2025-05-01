import torch


def compute_kernel_matrix(X, Y, kernel_config, device, dtype, kernel_func):
    """
    General function to compute kernel matrices between X and Y.

    Args:
        X: First set of points (n_x, d)
        Y: Second set of points (n_y, d)
        kernel_config: Kernel configuration containing lengthscale and const_scaling
        device: Device for the output
        dtype: Data type for the output
        kernel_func: Function that computes the kernel value between two vectors

    Returns:
        K: Kernel matrix (n_x, n_y)
    """
    K = torch.zeros((X.shape[0], Y.shape[0]), device=device, dtype=dtype)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i, j] = kernel_config.const_scaling * kernel_func(
                X[i], Y[j], kernel_config.lengthscale
            )
    return K


def rbf_kernel(x, y, lengthscale):
    """Compute RBF kernel between two vectors."""
    return torch.exp(-1 / 2 * torch.sum(((x - y) / lengthscale) ** 2))


def laplace_kernel(x, y, lengthscale):
    """Compute Laplace kernel between two vectors."""
    return torch.exp(-torch.sum(torch.abs(x - y) / lengthscale))


def _distance_matrix_matern(x, y, lengthscale):
    """Compute scaled distance matrix for Matern kernels."""
    return torch.sum(((x - y) / lengthscale) ** 2) ** 0.5


def matern12_kernel(x, y, lengthscale):
    """Compute Matern-1/2 kernel between two vectors."""
    d = _distance_matrix_matern(x, y, lengthscale)
    return torch.exp(-d)


def matern32_kernel(x, y, lengthscale):
    """Compute Matern-3/2 kernel between two vectors."""
    d = _distance_matrix_matern(x, y, lengthscale)
    sqrt3 = 3**0.5
    return (1 + sqrt3 * d) * torch.exp(-sqrt3 * d)


def matern52_kernel(x, y, lengthscale):
    """Compute Matern-5/2 kernel between two vectors."""
    d = _distance_matrix_matern(x, y, lengthscale)
    sqrt5 = 5**0.5
    return (1 + sqrt5 * d + 5 / 3 * d**2) * torch.exp(-sqrt5 * d)
