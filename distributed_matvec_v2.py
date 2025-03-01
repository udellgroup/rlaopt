import torch
import time

from rlaopt.utils import LinOp
from pykeops.torch import LazyTensor


def get_rbf_linop(Ab_lazy, A_chunk, blk_sz, sigma=1.0):
    A_chunk_lazy = LazyTensor(A_chunk[None, :, :])
    D = ((Ab_lazy - A_chunk_lazy) ** 2).sum(dim=2)
    K = (-D / (2 * sigma**2)).exp()

    def matvec(v):
        return K @ v

    return LinOp(shape=(blk_sz, A_chunk.shape[0]), matvec=matvec)


# def distributed_matvec(K_chunks, x_chunks, streams, results, final_result):
#     for (K_chunk, x_chunk, stream, result) in zip(K_chunks, x_chunks,
#       streams, results):
#         with torch.cuda.stream(stream):
#             result.copy_(K_chunk @ x_chunk)

#     # Using preallocated final_result
#     final_result.zero_()  # Reset the final result tensor to zero

#     final_device = final_result.device
#     with torch.cuda.stream(streams[0]):
#         for result in results:
#             final_result.add_(result.to(final_device))

#     # Synchronize all streams
#     torch.cuda.synchronize()

#     return final_result


def distributed_matvec(K_chunks, x_chunks, streams, results, final_result):
    events = []

    # Launch matrix multiplications in separate streams
    for (K_chunk, x_chunk, stream, result) in zip(K_chunks, x_chunks, streams, results):
        with torch.cuda.stream(stream):
            result.copy_(K_chunk @ x_chunk)
            event = torch.cuda.Event(enable_timing=False)
            event.record(stream)
            events.append(event)

    # Using preallocated final_result
    final_result.zero_()  # Reset the final result tensor to zero

    final_device = final_result.device
    main_stream = torch.cuda.current_stream()

    # Wait for all chunk computations to complete before summing
    for event in events:
        event.wait(main_stream)

    # Sum the results in the main stream
    with torch.cuda.stream(main_stream):
        for result in results:
            final_result.add_(result.to(final_device))

    # Create and record a final event
    final_event = torch.cuda.Event(enable_timing=False)
    final_event.record(main_stream)

    return final_result, final_event


torch.set_default_dtype(torch.float32)

# Ensure CUDA is available and get the device count
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script requires a machine with CUDA-enabled GPUs."
    )
device_count = torch.cuda.device_count()
print(f"CUDA Device Count: {device_count}")

# Initialize a large matrix A and vector x
n = 500000
d = 500
blk_sz = 10000  # block size
matrix_size = (n, d)
vector_size = (n,)

A = torch.randn(matrix_size).cuda(0)
A /= n**0.5
x = torch.randn(vector_size).cuda(0)
x /= d**0.5
blk = torch.multinomial(torch.ones(n), blk_sz, replacement=False)

# Split matrix A to distribute across the GPUs (by rows)
A_chunks = [
    chunk.to(f"cuda:{i}") for i, chunk in enumerate(torch.chunk(A, device_count, dim=0))
]
# Put the block matrix on all GPUs
Ab_lazy_chunks = [
    LazyTensor((A[blk].to(f"cuda:{i}"))[:, None, :]) for i in range(device_count)
]
# Split vector x to distribute across the GPUs (by elements)
x_chunks = [
    chunk.to(f"cuda:{i}") for i, chunk in enumerate(torch.chunk(x, device_count, dim=0))
]
# Get chunked kernel linear operators
K_chunks = [
    get_rbf_linop(Ab_lazy_chunk, A_chunk, blk_sz)
    for Ab_lazy_chunk, A_chunk in zip(Ab_lazy_chunks, A_chunks)
]

# Create CUDA streams for parallel computations
streams = [torch.cuda.Stream(device=i) for i in range(device_count)]

# Preallocate result tensors outside of the function
preallocated_results = [
    torch.zeros(blk_sz, device=A_chunk.device, dtype=A_chunk.dtype)
    for A_chunk in A_chunks
]
final_result = torch.zeros(blk_sz, device=A_chunks[0].device, dtype=A_chunks[0].dtype)

# Perform the distributed matrix-vector multiplication
ts = time.time()
result, final_event = distributed_matvec(
    K_chunks, x_chunks, streams, preallocated_results, final_result
)
final_event.synchronize()
print(f"Elapsed time for distributed matvec = {time.time() - ts}")

# Verify the result with the non-distributed version
ts = time.time()
K = get_rbf_linop(LazyTensor((A[blk].to("cuda:0"))[:, None, :]), A, blk_sz)
expected_result = K @ x
print(f"Elapsed time for non-distributed matvec = {time.time() - ts}")
print(f"Results are close: {torch.allclose(result, expected_result)}")
print(f"Max difference: {torch.abs(result - expected_result).max()}")
