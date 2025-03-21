import torch
import torch.multiprocessing as mp
from rlaopt.linops import LinOp
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


def initial_worker(device_id):
    torch.cuda.set_device(device_id)


def worker(task):
    blk_sz, Ab_lazy_slice, A_chunk_slice, x_chunk_slice, device_id = task
    device = f"cuda:{device_id}"

    try:
        Ab_lazy_chunk = LazyTensor(Ab_lazy_slice[:, None, :].to(device))
        A_chunk_lazy = LazyTensor(A_chunk_slice[None, :, :])

        K_chunk = RBFLinearOperator(Ab_lazy_chunk, A_chunk_lazy, blk_sz)
        result = K_chunk.matvec(x_chunk_slice)

        return result.cpu()
    except Exception as e:
        print(f"Error in worker on device {device_id}: {e}")
        return None


def main():
    torch.set_default_dtype(torch.float32)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script requires a machine with \
                CUDA-enabled GPUs."
        )

    # Initialize data
    n, d, blk_sz = 500000, 500, 50000
    matrix_size, vector_size = (n, d), (n,)
    A = torch.randn(matrix_size).cuda()
    A /= n**0.5
    x = torch.randn(vector_size).cuda()
    x /= d**0.5

    device_count = torch.cuda.device_count()
    print(f"CUDA Device Count: {device_count}")

    # Split the matrix and vector for distributing across GPUs
    A_chunks = [
        chunk.to(f"cuda:{i}")
        for i, chunk in enumerate(torch.chunk(A, device_count, dim=0))
    ]
    x_chunks = [
        chunk.to(f"cuda:{i}")
        for i, chunk in enumerate(torch.chunk(x, device_count, dim=0))
    ]

    # Initialize pool of workers
    pool = mp.Pool(processes=device_count, initializer=initial_worker, initargs=(0,))

    num_operations = 10  # Example number of matrix-vector products

    for operation in range(num_operations):
        blk = torch.multinomial(torch.ones(n), blk_sz, replacement=False)

        tasks = []
        for i in range(device_count):
            Ab_lazy_slice = A[blk]
            x_chunk_slice = x_chunks[i]
            task = (blk_sz, Ab_lazy_slice, A_chunks[i], x_chunk_slice, i)
            tasks.append(task)

        preallocated_results = pool.map(worker, tasks)

        final_result = torch.zeros(blk_sz, device="cuda:0")
        for result in preallocated_results:
            if result is not None:
                final_result.add_(result.to("cuda:0"))

        # Check results against the expected non-distributed version
        # for a single example
        K = get_rbf_linop(
            LazyTensor(A[blk][:, None, :].to("cuda:0")),
            LazyTensor(A[None, :, :].to("cuda:0")),
            blk_sz,
        )
        expected_result = K.matvec(x.to("cuda:0"))
        print(f"Results are close: {torch.allclose(final_result, expected_result)}")
        print(f"Max difference: {torch.abs(final_result - expected_result).max()}")

    # Shutdown the pool
    pool.close()
    pool.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
