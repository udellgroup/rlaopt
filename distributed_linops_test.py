from functools import partial

import torch
from torch.multiprocessing import Pool, set_start_method
from typing import List

from rlaopt.linops import (
    LinOp,
    TwoSidedLinOp,
    DistributedLinOp,
    DistributedTwoSidedLinOp,
)


def matvec(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix @ x


def rmatvec(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix.T @ x


# Helper function to initialize worker devices
def initialize_worker(device_id: int, n_devices: int):
    if torch.cuda.is_available():
        device = device_id % n_devices
        torch.cuda.set_device(device)
    else:
        print(f"Worker {device_id} using CPU")


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

    # Create the pool with device initializers
    pool = Pool(
        processes=num_workers,
        initializer=initialize_worker,
        initargs=(list(range(num_workers))[0], n_devices),
    )

    sizes = torch.arange(1, num_workers + 1) * 1000

    # Example matrices for chunks
    matrices = [
        torch.rand(sz, 10000).to(
            f"cuda:{i % n_devices}" if torch.cuda.is_available() else "cpu"
        )
        for i, sz in zip(range(num_workers), sizes)
    ]

    # Copy the matrices and put them on the same device
    matrices_same_device = [mat.to(matrices[0].device) for mat in matrices]

    # Create linop chunks
    linop_chunks = create_linop_chunks(matrices)

    # Initialize the distributed linear operator
    shape = torch.Size((sum(mat.shape[0] for mat in matrices), matrices[0].shape[1]))
    # dist_lin_op = DistributedLinOp(shape=shape, A=linop_chunks)
    dist_lin_op = DistributedLinOp(shape=shape, A=linop_chunks, pool=pool)

    # Test with a vector
    vector = torch.rand(shape[1]).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = dist_lin_op @ vector
    result_true = torch.cat(matrices_same_device, dim=0) @ vector
    print("Are results equal?", torch.allclose(result, result_true))

    # Test with a matrix
    matrix = torch.rand(shape[1], 2).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_lin_op @ matrix
    result_true = torch.cat(matrices_same_device, dim=0) @ matrix
    print("Are results equal?", torch.allclose(result, result_true))

    # Create twosided linop chunks
    twosided_linop_chunks = create_twosided_linop_chunks(matrices)

    # Initialize the distributed two-sided linear operator
    shape = torch.Size((sum(mat.shape[0] for mat in matrices), matrices[0].shape[1]))
    dist_twosided_lin_op = DistributedTwoSidedLinOp(
        shape=shape, A=twosided_linop_chunks, pool=pool
    )

    # Test with a vector
    vector = torch.rand(shape[0]).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = vector @ dist_twosided_lin_op
    result_true = (
        vector.unsqueeze(0) @ torch.cat(matrices_same_device, dim=0)
    ).squeeze(0)
    print("Are results equal?", torch.allclose(result, result_true))

    # Test with a matrix
    matrix = torch.rand(2, shape[0]).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = matrix @ dist_twosided_lin_op
    result_true = matrix @ torch.cat(matrices_same_device, dim=0)
    print("Are results equal?", torch.allclose(result, result_true))

    # Shutdown the pool
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
