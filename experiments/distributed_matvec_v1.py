import torch
import time


# Define the distributed matvec function with CUDA timing
def distributed_matvec(A_chunks, x_chunks, streams, results):
    # Launch asynchronous matvec operations
    for i, (A_chunk, x_chunk, stream, result) in enumerate(
        zip(A_chunks, x_chunks, streams, results)
    ):
        with torch.cuda.stream(stream):
            # result.copy_((torch.matmul(A_chunk.to(torch.float64), \
            # x_chunk.to(torch.float64))).to(torch.float32))
            result.copy_(torch.matmul(A_chunk, x_chunk))

    # Synchronize all streams
    torch.cuda.synchronize()

    # Gather results from all devices
    results_cpu = [result.cpu() for result in results]

    # Concatenate the results
    final_result = torch.cat(results_cpu, dim=0).cuda(0)

    return final_result


torch.set_default_dtype(torch.float32)

# Ensure CUDA is available and get the device count
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script requires a machine with CUDA-enabled GPUs."
    )
device_count = torch.cuda.device_count()
print(f"CUDA Device Count: {device_count}")

# Initialize a large matrix A and vector x
n = 120000
d = 100000
matrix_size = (n, d)
vector_size = (d,)

A = torch.randn(matrix_size).cuda(0)
A /= n**0.5
x = torch.randn(vector_size).cuda(0)
x /= d**0.5

# Split matrix A to distribute across the GPUs (by rows)
A_chunks = [
    chunk.to(f"cuda:{i}") for i, chunk in enumerate(torch.chunk(A, device_count, dim=0))
]
x_chunks = [
    x.to(f"cuda:{i}") for i in range(device_count)
]  # Pre-distribute x across GPUs

# Create CUDA streams for parallel computations
streams = [torch.cuda.Stream(device=i) for i in range(device_count)]

# Preallocate result tensors outside of the function
preallocated_results = [
    torch.zeros(A_chunk.shape[0], device=f"cuda:{i}", dtype=A_chunk.dtype)
    for i, A_chunk in enumerate(A_chunks)
]

# Perform the distributed matrix-vector multiplication
ts = time.time()
result = distributed_matvec(A_chunks, x_chunks, streams, preallocated_results)
print(f"Elapsed time for distributed matvec = {time.time() - ts}")

# Verify the result with the non-distributed version
ts = time.time()
expected_result = A @ x
print(f"Elapsed time for non-distributed matvec = {time.time() - ts}")
print(f"Results are close: {torch.allclose(result, expected_result)}")
print(f"Max difference: {torch.abs(result - expected_result).max()}")
