import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rlaopt.utils import LinOp
from pykeops.torch import LazyTensor


class RBFLinearOperator(LinOp):
    def __init__(self, Ab_lazy, A_chunk_lazy, blk_sz, sigma=1.0):
        self.A_chunk_lazy = A_chunk_lazy
        self.Ab_lazy = Ab_lazy
        self.sigma = sigma
        self.blk_sz = blk_sz
        self._shape = (blk_sz, A_chunk_lazy.shape[0])

        # Compute the kernel matrix
        D = ((self.Ab_lazy - self.A_chunk_lazy) ** 2).sum(dim=2)
        self.K = (-D / (2 * self.sigma**2)).exp()

    def matvec(self, v):
        with torch.no_grad():
            return self.K @ v


def get_rbf_linop(Ab_lazy, A_chunk, blk_sz, sigma=1.0):
    return RBFLinearOperator(Ab_lazy, A_chunk, blk_sz, sigma)


def worker(rank, world_size, A, x, blk_indices_list):
    os.environ[
        "MASTER_ADDR"
    ] = "127.0.0.1"  # Replace with master node IP in multi-node setup
    os.environ["MASTER_PORT"] = "29500"  # Choose a port that is open

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    for trial, blk_indices in enumerate(blk_indices_list, start=1):
        blk_sz = blk_indices.shape[0]  # Determine blk_sz dynamically

        Ab_lazy_slice = A[blk_indices].to(device)
        A_chunk_slice = A.chunk(world_size, dim=0)[rank].to(device)
        x_chunk_slice = x.chunk(world_size, dim=0)[rank].to(device)

        K_chunk = RBFLinearOperator(
            LazyTensor(Ab_lazy_slice[:, None, :]),
            LazyTensor(A_chunk_slice[None, :, :]),
            blk_sz,
        )  # No DDP since the module has no parameters

        result = K_chunk.matvec(x_chunk_slice)

        # Gather results from all ranks
        gathered_results = [
            torch.zeros(blk_sz, device=device, dtype=A.dtype) for _ in range(world_size)
        ]
        dist.all_gather(gathered_results, result)

        if rank == 0:
            final_result = torch.stack(gathered_results).sum(dim=0)

            # Check results against expected non-distributed version
            # for a single example
            K = get_rbf_linop(
                LazyTensor(A[blk_indices][:, None, :].to("cuda:0")),
                LazyTensor(A[None, :, :].to("cuda:0")),
                blk_sz,
            )
            expected_result = K.matvec(x.to("cuda:0"))
            print(
                "Results are close: "
                f"{torch.allclose(final_result, expected_result, atol=1e-4)}"
            )
            print(f"Max difference: {torch.abs(final_result - expected_result).max()}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This script requires a machine with CUDA-enabled GPUs."
        )

    n, d, blk_sz = 500000, 500, 50000
    matrix_size, vector_size = (n, d), (n,)
    A = torch.randn(matrix_size, dtype=dtype).cuda()
    A /= n**0.5
    x = torch.randn(vector_size, dtype=dtype).cuda()
    x /= d**0.5

    world_size = torch.cuda.device_count()

    # Sample block indices for 10 trials in the main process
    blk_indices_list = [
        torch.multinomial(torch.ones(n), blk_sz, replacement=False) for _ in range(10)
    ]

    # Spawn workers with the sampled block indices for each trial
    mp.spawn(
        worker, args=(world_size, A, x, blk_indices_list), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    main()
