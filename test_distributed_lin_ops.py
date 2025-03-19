from functools import partial

import torch
from torch.multiprocessing import Pool, set_start_method
from typing import Tuple, List, Callable

from rlaopt.utils.linops import (
    LinOp,
    TwoSidedLinOp,
    DistributedLinOp,
    DistributedTwoSidedLinOp,
    DistributedSymmetricLinOp,
)


def matvec(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix @ x


# Helper function to initialize worker devices
def initialize_worker(device_id: int, n_devices: int):
    if torch.cuda.is_available():
        device = device_id % n_devices
        torch.cuda.set_device(device)
    else:
        print(f"Worker {device_id} using CPU")


def create_matvec(matrix: torch.Tensor) -> Callable:
    def matvec(x: torch.Tensor):
        return matrix @ x

    return matvec


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

    # Example matrices for chunks
    matrices = [
        torch.rand(10000, 10000).to(
            f"cuda:{i % n_devices}" if torch.cuda.is_available() else "cpu"
        )
        for i in range(num_workers)
    ]

    # Create linop chunks
    linop_chunks = create_linop_chunks(matrices)

    # Initialize the distributed linear operator
    shape = (sum(mat.shape[0] for mat in matrices), matrices[0].shape[1])
    dist_lin_op = DistributedLinOp(shape=shape, A=linop_chunks)

    # Test with a vector
    vector = torch.rand(shape[1]).to("cuda:0" if torch.cuda.is_available() else "cpu")
    result = dist_lin_op @ (pool, vector)
    result_true = torch.cat(
        [
            (linop_chunks[i] @ vector.to(linop_chunks[i].device)).to(vector.device)
            for i in range(num_workers)
        ],
        dim=0,
    )
    print("Are results equal?", torch.allclose(result, result_true))

    # Test with a matrix
    matrix = torch.rand(shape[1], 2).to(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = dist_lin_op @ (pool, matrix)
    result_true = torch.cat(
        [
            (linop_chunks[i] @ matrix.to(linop_chunks[i].device)).to(matrix.device)
            for i in range(num_workers)
        ],
        dim=0,
    )
    print("Are results equal?", torch.allclose(result, result_true))

    # Shutdown the pool
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
