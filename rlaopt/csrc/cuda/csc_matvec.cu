#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace rlaopt {

namespace {
// Get properties of the current CUDA device
cudaDeviceProp get_device_properties() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props;
}

// Helper struct to store device grid limits
struct DeviceGridLimits {
    int max_grid_dim_x;
};

// Helper to get device maximum grid dimension
DeviceGridLimits get_device_grid_limits(const cudaDeviceProp& props) {
    DeviceGridLimits limits;
    limits.max_grid_dim_x = props.maxGridSize[0];
    return limits;
}

// Get optimal thread block configuration for vector operations
int get_optimal_block_size(const cudaDeviceProp& props) {
    int max_threads_per_block = props.maxThreadsPerBlock;
    int warp_size = props.warpSize;

    // Default to 256 threads per block for good occupancy
    int block_size = 256;

    // Adjust based on device capabilities
    if (max_threads_per_block < 1024) {
        block_size = max_threads_per_block / 2;
    }

    // Ensure block_size is a multiple of warp_size
    block_size = (block_size / warp_size) * warp_size;

    return block_size;
}

// CUDA kernel for CSC matrix-vector product
template <typename scalar_t>
__global__ void csc_matvec_kernel(const scalar_t* __restrict__ values,
                                  const int64_t* __restrict__ row_indices,
                                  const int64_t* __restrict__ col_ptrs,
                                  const scalar_t* __restrict__ dense_vector,
                                  scalar_t* __restrict__ result, const int64_t num_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_cols) {
        scalar_t x_j = dense_vector[col];

        for (int64_t k = col_ptrs[col]; k < col_ptrs[col + 1]; ++k) {
            int64_t row = row_indices[k];
            scalar_t value = values[k];
            atomicAdd(&result[row], value * x_j);
        }
    }
}
}  // namespace

torch::Tensor csc_matvec_cuda(const torch::Tensor& sparse_tensor,
                              const torch::Tensor& dense_vector) {
    TORCH_CHECK(sparse_tensor.layout() == at::kSparseCsc, "Input tensor must be in CSC format");
    TORCH_CHECK(dense_vector.is_contiguous(), "dense_vector must be contiguous");
    TORCH_CHECK(dense_vector.dim() == 1, "dense_vector must be 1-dimensional");
    TORCH_CHECK(sparse_tensor.device().type() == at::DeviceType::CUDA,
                "Input tensor must be on CUDA");
    TORCH_CHECK(dense_vector.device().type() == at::DeviceType::CUDA,
                "dense_vector must be on CUDA");

    TORCH_CHECK(sparse_tensor.dtype() == dense_vector.dtype(),
                "sparse_tensor and dense_vector must have the same dtype");
    TORCH_CHECK(sparse_tensor.dtype() == torch::kFloat || sparse_tensor.dtype() == torch::kDouble,
                "sparse_tensor must be float32 or float64");

    auto values = sparse_tensor.values();
    auto row_indices = sparse_tensor.row_indices();
    auto col_ptrs = sparse_tensor.ccol_indices();

    auto num_rows = sparse_tensor.size(0);
    auto num_cols = sparse_tensor.size(1);

    TORCH_CHECK(num_cols == dense_vector.size(0),
                "Number of columns in sparse tensor must match dense vector size");

    auto result = torch::zeros({num_rows}, dense_vector.options());

    // Get device properties
    cudaDeviceProp props = get_device_properties();

    // Get optimal block size based on device capabilities
    int threads_per_block = get_optimal_block_size(props);

    // Get maximum grid dimension from device properties
    DeviceGridLimits grid_limits = get_device_grid_limits(props);
    int64_t MAX_GRID_DIM_X = grid_limits.max_grid_dim_x;

    // Process the matrix in column chunks if needed
    for (int64_t col_start = 0; col_start < num_cols;
         col_start += MAX_GRID_DIM_X * threads_per_block) {
        int64_t cols_in_chunk = std::min(MAX_GRID_DIM_X * threads_per_block, num_cols - col_start);
        int num_blocks = (cols_in_chunk + threads_per_block - 1) / threads_per_block;

        // // Uncomment for debugging
        // printf("Processing columns %ld to %ld (Blocks: %d, Threads per block: %d)\n",
        //        col_start, col_start + cols_in_chunk - 1, num_blocks, threads_per_block);

        AT_DISPATCH_FLOATING_TYPES(
            sparse_tensor.scalar_type(), "csc_matvec_cuda", ([&] {
                csc_matvec_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                    values.data_ptr<scalar_t>(), row_indices.data_ptr<int64_t>(),
                    col_ptrs.data_ptr<int64_t>() + col_start,
                    dense_vector.data_ptr<scalar_t>() + col_start, result.data_ptr<scalar_t>(),
                    cols_in_chunk);
            }));
    }

    return result;
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCUDA, m) { m.impl("csc_matvec", &csc_matvec_cuda); }

}  // namespace rlaopt
