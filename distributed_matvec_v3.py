import torch
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
        return self.K @ v


def get_rbf_linop(Ab_lazy, A_chunk, blk_sz, sigma=1.0):
    return RBFLinearOperator(Ab_lazy, A_chunk, blk_sz, sigma)


def worker(rank, blk_sz, Ab_lazy_slice, A_chunk_slice, x_chunk_slice, result_queue):
    # Initialize CUDA environment in the worker
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # Explicitly reinitialize LazyTensor variables within the worker
    Ab_lazy_chunk = LazyTensor(Ab_lazy_slice[:, None, :].to(device))
    A_chunk_lazy = LazyTensor(A_chunk_slice[None, :, :].to(device))

    # Create the RBFLinearOperator in the worker
    try:
        K_chunk = RBFLinearOperator(Ab_lazy_chunk, A_chunk_lazy, blk_sz)
        result = K_chunk.matvec(x_chunk_slice.to(device))
        result_queue.put(result.cpu())
    except Exception as e:
        print(f"Error in worker on device {rank}: {e}")
        result_queue.put(None)


def main():
    torch.set_default_dtype(torch.float32)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script requires a machine with \
                CUDA-enabled GPUs."
        )

    device_count = torch.cuda.device_count()
    print(f"CUDA Device Count: {device_count}")

    # Initialize large matrix A and vector x
    n, d, blk_sz = 500000, 500, 10000  # block size
    matrix_size, vector_size = (n, d), (n,)

    A = torch.randn(matrix_size).cuda()
    A /= n**0.5
    x = torch.randn(vector_size).cuda()
    x /= d**0.5
    blk = torch.multinomial(torch.ones(n), blk_sz, replacement=False)

    # Split matrix A and vector x to distribute across GPUs
    A_chunks = [
        chunk.to(f"cuda:{i}")
        for i, chunk in enumerate(torch.chunk(A, device_count, dim=0))
    ]
    Ab_lazy_slices = [A[blk].to(f"cuda:{i}") for i in range(device_count)]
    x_chunks = [
        chunk.to(f"cuda:{i}")
        for i, chunk in enumerate(torch.chunk(x, device_count, dim=0))
    ]

    # Multiprocessing queue to collect results
    result_queue = mp.Queue()

    # Create and start processes
    processes = []
    for rank, (Ab_lazy_slice, A_chunk, x_chunk) in enumerate(
        zip(Ab_lazy_slices, A_chunks, x_chunks)
    ):
        p = mp.Process(
            target=worker,
            args=(rank, blk_sz, Ab_lazy_slice, A_chunk, x_chunk, result_queue),
        )
        p.start()
        processes.append(p)

    # Collect results from processes
    preallocated_results = []
    for _ in range(device_count):
        result = result_queue.get()
        if result is not None:
            preallocated_results.append(result)
        else:
            raise RuntimeError("A worker process failed.")

    for p in processes:
        p.join()

    # Aggregate results in the main process
    final_result = torch.zeros(blk_sz, device="cuda:0")
    for result in preallocated_results:
        final_result.add_(result.to("cuda:0"))

    # Verification with non-distributed version
    K = get_rbf_linop(
        LazyTensor(A[blk][:, None, :].to("cuda:0")),
        LazyTensor(A[None, :, :].to("cuda:0")),
        blk_sz,
    )
    expected_result = K.matvec(x.to("cuda:0"))
    print(f"Results are close: {torch.allclose(final_result, expected_result)}")
    print(f"Max difference: {torch.abs(final_result - expected_result).max()}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
