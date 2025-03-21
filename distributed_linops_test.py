from functools import partial

import torch
from torch.multiprocessing import set_start_method
from typing import List

from rlaopt.linops import (
    LinOp,
    TwoSidedLinOp,
    DistributedLinOp,
    DistributedTwoSidedLinOp,
    DistributionMode,
)


def matvec(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix @ x


def rmatvec(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix.T @ x


def create_linop_chunks(matrices: List[torch.Tensor]) -> List[LinOp]:
    return [
        LinOp(
            matrix.device,
            matrix.shape,
            partial(matvec, matrix=matrix),
            partial(matvec, matrix=matrix),
        )
        for matrix in matrices
    ]


def create_twosided_linop_chunks(matrices: List[torch.Tensor]) -> List[TwoSidedLinOp]:
    return [
        TwoSidedLinOp(
            matrix.device,
            matrix.shape,
            partial(matvec, matrix=matrix),
            partial(rmatvec, matrix=matrix),
            partial(matvec, matrix=matrix),
            partial(rmatvec, matrix=matrix),
        )
        for matrix in matrices
    ]


def main():
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_workers = 4  # Modify based on the number of chunks and available devices

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ---- Test Row-Distributed Case ----
    print("\n==== Testing Row-Distributed Case ====")

    # Create random matrices of different sizes (varying rows)
    row_sizes = torch.arange(1, num_workers + 1) * 1000
    col_size = 10000

    row_matrices = [
        torch.rand(sz, col_size).to(
            f"cuda:{i % n_devices}" if torch.cuda.is_available() else "cpu"
        )
        for i, sz in zip(range(num_workers), row_sizes)
    ]

    # Copy the matrices and put them on the same device for comparison
    row_matrices_same_device = [mat.to(row_matrices[0].device) for mat in row_matrices]
    row_combined_matrix = torch.cat(row_matrices_same_device, dim=0)

    # Create linop chunks
    row_linop_chunks = create_linop_chunks(row_matrices)

    # Test LinOp with row distribution
    row_shape = torch.Size((sum(mat.shape[0] for mat in row_matrices), col_size))
    dist_row_lin_op = DistributedLinOp(
        shape=row_shape, A=row_linop_chunks, distribution_mode=DistributionMode.ROW
    )

    # Test with a vector
    vector = torch.rand(col_size).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = dist_row_lin_op @ vector
    result_true = row_combined_matrix @ vector
    print("LinOp Row Distributed - Vector: ", torch.allclose(result, result_true))

    # Test with a matrix
    matrix = torch.rand(col_size, 2).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_row_lin_op @ matrix
    result_true = row_combined_matrix @ matrix
    print("LinOp Row Distributed - Matrix: ", torch.allclose(result, result_true))

    # Test TwoSidedLinOp with row distribution
    row_twosided_linop_chunks = create_twosided_linop_chunks(row_matrices)
    dist_row_twosided_lin_op = DistributedTwoSidedLinOp(
        shape=row_shape,
        A=row_twosided_linop_chunks,
        distribution_mode=DistributionMode.ROW,
    )

    # Test forward with a vector
    vector = torch.rand(col_size).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = dist_row_twosided_lin_op @ vector
    result_true = row_combined_matrix @ vector
    print(
        "TwoSidedLinOp Row Distributed - Forward Vector: ",
        torch.allclose(result, result_true),
    )

    # Test transposed with a vector
    vector = torch.rand(row_shape[0]).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_row_twosided_lin_op.T @ vector
    result_true = row_combined_matrix.T @ vector
    print(
        "TwoSidedLinOp Row Distributed - Transpose Vector: ",
        torch.allclose(result, result_true),
    )

    # Clean up
    del dist_row_lin_op
    del dist_row_twosided_lin_op

    # ---- Test Column-Distributed Case ----
    print("\n==== Testing Column-Distributed Case ====")

    # Create random matrices with varying column sizes
    row_size = 10000
    col_sizes = torch.arange(1, num_workers + 1) * 1000

    col_matrices = [
        torch.rand(row_size, sz).to(
            f"cuda:{i % n_devices}" if torch.cuda.is_available() else "cpu"
        )
        for i, sz in zip(range(num_workers), col_sizes)
    ]

    # Copy the matrices and put them on the same device for comparison
    col_matrices_same_device = [mat.to(col_matrices[0].device) for mat in col_matrices]
    col_combined_matrix = torch.cat(col_matrices_same_device, dim=1)

    # Create linop chunks for column distribution
    col_linop_chunks = create_linop_chunks(col_matrices)

    # Test LinOp with column distribution
    col_shape = torch.Size((row_size, sum(mat.shape[1] for mat in col_matrices)))
    dist_col_lin_op = DistributedLinOp(
        shape=col_shape, A=col_linop_chunks, distribution_mode=DistributionMode.COLUMN
    )

    # Test with a vector
    vector = torch.rand(col_shape[1]).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_col_lin_op @ vector
    result_true = col_combined_matrix @ vector
    print("LinOp Column Distributed - Vector: ", torch.allclose(result, result_true))

    # Test with a matrix
    matrix = torch.rand(col_shape[1], 2).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_col_lin_op @ matrix
    result_true = col_combined_matrix @ matrix
    print("LinOp Column Distributed - Matrix: ", torch.allclose(result, result_true))

    # Test TwoSidedLinOp with column distribution
    col_twosided_linop_chunks = create_twosided_linop_chunks(col_matrices)
    dist_col_twosided_lin_op = DistributedTwoSidedLinOp(
        shape=col_shape,
        A=col_twosided_linop_chunks,
        distribution_mode=DistributionMode.COLUMN,
    )

    # Test forward with a vector
    vector = torch.rand(col_shape[1]).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_col_twosided_lin_op @ vector
    result_true = col_combined_matrix @ vector
    print(
        "TwoSidedLinOp Column Distributed - Forward Vector: ",
        torch.allclose(result, result_true),
    )

    # Test right multiplication with a vector (v @ A)
    vector = torch.rand(row_size).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = vector @ dist_col_twosided_lin_op
    result_true = vector @ col_combined_matrix
    print(
        "TwoSidedLinOp Column Distributed - Right Mult Vector: ",
        torch.allclose(result, result_true),
    )

    # Test transposed with a vector
    vector = torch.rand(row_size).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = dist_col_twosided_lin_op.T @ vector
    result_true = col_combined_matrix.T @ vector
    print(
        "TwoSidedLinOp Column Distributed - Transpose Vector: ",
        torch.allclose(result, result_true),
    )

    # Test transposed right multiplication with a vector (v @ A.T)
    vector = torch.rand(col_shape[1]).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = vector @ dist_col_twosided_lin_op.T
    result_true = vector @ col_combined_matrix.T
    print(
        "TwoSidedLinOp Column Distributed - Transpose Right Mult: ",
        torch.allclose(result, result_true),
    )

    # Clean up
    del dist_col_lin_op
    del dist_col_twosided_lin_op

    # ---- Test Switching Distribution Modes via Transpose ----
    print("\n==== Testing Distribution Mode Switching via Transpose ====")

    # Start with row-distributed
    dist_row_twosided_lin_op = DistributedTwoSidedLinOp(
        shape=row_shape,
        A=row_twosided_linop_chunks,
        distribution_mode=DistributionMode.ROW,
    )

    # Get its transpose (which should be column-distributed)
    dist_col_via_transpose = dist_row_twosided_lin_op.T

    # Test with a vector
    vector = torch.rand(row_shape[0]).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_col_via_transpose @ vector
    result_true = row_combined_matrix.T @ vector
    print(
        "Row → Column via Transpose - Forward Vector: ",
        torch.allclose(result, result_true),
    )

    # Now get back to row-distributed via another transpose
    dist_row_via_transpose = dist_col_via_transpose.T

    # Test with original vector
    vector = torch.rand(col_size).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = dist_row_via_transpose @ vector
    result_true = row_combined_matrix @ vector
    print(
        "Column → Row via Transpose - Forward Vector: ",
        torch.allclose(result, result_true),
    )

    # Clean up
    del dist_row_twosided_lin_op


if __name__ == "__main__":
    main()
