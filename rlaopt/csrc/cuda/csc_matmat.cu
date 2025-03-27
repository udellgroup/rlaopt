#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace rlaopt {

// CUDA kernel for CSC matrix-matrix product with 2D thread blocks
template <typename scalar_t>
__global__ void csc_matmat_kernel_2d(
    const scalar_t* __restrict__ values, const int64_t* __restrict__ row_indices,
    const int64_t* __restrict__ col_ptrs, const scalar_t* __restrict__ dense_matrix,
    scalar_t* __restrict__ result, int64_t num_cols, int64_t batch_size, int64_t dense_col_stride,
    int64_t dense_batch_stride, int64_t result_row_stride, int64_t result_batch_stride) {
    // x dimension is for columns and y dimension is for batches
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < num_cols && b < batch_size) {
        scalar_t x_jb = dense_matrix[col * dense_col_stride + b * dense_batch_stride];

        // Skip computation if the value is zero
        if (x_jb == 0) return;

        for (int64_t k = col_ptrs[col]; k < col_ptrs[col + 1]; ++k) {
            int64_t row = row_indices[k];
            scalar_t value = values[k];
            atomicAdd(&result[row * result_row_stride + b * result_batch_stride], value * x_jb);
        }
    }
}

// Helper struct to store device grid limits
struct DeviceGridLimits {
    int max_grid_dim_x;
    int max_grid_dim_y;
};

// Helper to get device properties and maximum grid dimensions
DeviceGridLimits get_device_grid_limits() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    DeviceGridLimits limits;
    limits.max_grid_dim_x = props.maxGridSize[0];
    limits.max_grid_dim_y = props.maxGridSize[1];

    return limits;
}

// Optimized configuration for when the sparse matrix has more columns than the batch size
dim3 get_optimal_block_config(int64_t batch_size) {
    // We aim for ~256 threads per block for good occupancy
    const int target_threads = 256;

    // Start with default 1x256 configuration - optimized for column-heavy matrices
    int threads_x = 1;    // For columns now (just 1 column per thread)
    int threads_y = 256;  // For batches now (process many batches per block)

    // If batch_size is small, adjust dimensions
    if (batch_size <= 128) {
        threads_y = batch_size;
        // Increase threads_x to maintain target thread count
        threads_x = target_threads / threads_y;
    } else if (batch_size < 256) {
        // If batch_size is between 129-255, just use the batch_size
        threads_y = batch_size;
        threads_x = 1;
    }

    return dim3(threads_x, threads_y);
}

torch::Tensor csc_matmat_cuda(const torch::Tensor& sparse_tensor,
                              const torch::Tensor& dense_matrix) {
    TORCH_CHECK(sparse_tensor.layout() == at::kSparseCsc, "Input tensor must be in CSC format");
    TORCH_CHECK(dense_matrix.dim() == 2, "dense_matrix must be 2-dimensional");
    TORCH_CHECK(sparse_tensor.device().type() == at::DeviceType::CUDA,
                "Input tensor must be on CUDA");
    TORCH_CHECK(dense_matrix.device().type() == at::DeviceType::CUDA,
                "dense_matrix must be on CUDA");

    TORCH_CHECK(sparse_tensor.dtype() == dense_matrix.dtype(),
                "sparse_tensor and dense_matrix must have the same dtype");
    TORCH_CHECK(sparse_tensor.dtype() == torch::kFloat || sparse_tensor.dtype() == torch::kDouble,
                "sparse_tensor must be float32 or float64");

    auto values = sparse_tensor.values();
    auto row_indices = sparse_tensor.row_indices();
    auto col_ptrs = sparse_tensor.ccol_indices();

    auto num_rows = sparse_tensor.size(0);
    auto num_cols = sparse_tensor.size(1);
    auto batch_size = dense_matrix.size(1);

    TORCH_CHECK(num_cols == dense_matrix.size(0),
                "Number of columns in sparse tensor must match dense matrix rows");

    auto result = torch::zeros({num_rows, batch_size}, dense_matrix.options());

    // Get strides for proper memory indexing
    auto dense_strides = dense_matrix.strides();
    auto result_strides = result.strides();

    int64_t dense_col_stride = dense_strides[0];
    int64_t dense_batch_stride = dense_strides[1];
    int64_t result_row_stride = result_strides[0];
    int64_t result_batch_stride = result_strides[1];

    // Dynamically determine optimal thread block configuration
    dim3 threads_per_block = get_optimal_block_config(batch_size);

    // Dynamically get maximum grid dimensions from the current device
    DeviceGridLimits grid_limits = get_device_grid_limits();
    int64_t MAX_GRID_DIM_X = grid_limits.max_grid_dim_x;
    int64_t MAX_GRID_DIM_Y = grid_limits.max_grid_dim_y;

    // Process the matrix in column chunks based on X-dimension limits (often larger)
    for (int64_t col_start = 0; col_start < num_cols;
         col_start += MAX_GRID_DIM_X * threads_per_block.x) {
        int64_t cols_in_chunk =
            std::min(MAX_GRID_DIM_X * threads_per_block.x, num_cols - col_start);
        int grid_x = (cols_in_chunk + threads_per_block.x - 1) / threads_per_block.x;

        // Process batches in chunks based on Y-dimension limits (often smaller)
        for (int64_t batch_start = 0; batch_start < batch_size;
             batch_start += MAX_GRID_DIM_Y * threads_per_block.y) {
            int64_t batches_in_chunk =
                std::min(MAX_GRID_DIM_Y * threads_per_block.y, batch_size - batch_start);
            int grid_y = (batches_in_chunk + threads_per_block.y - 1) / threads_per_block.y;

            dim3 num_blocks(grid_x, grid_y);

            // Uncomment for debugging
            printf(
                "Processing columns %lld to %lld, batches %lld to %lld (Grid: %d×%d, Block: "
                "%d×%d)\n",
                col_start, col_start + cols_in_chunk - 1, batch_start,
                batch_start + batches_in_chunk - 1, grid_x, grid_y, threads_per_block.x,
                threads_per_block.y);

            AT_DISPATCH_FLOATING_TYPES(
                sparse_tensor.scalar_type(), "csc_matmat_cuda", ([&] {
                    csc_matmat_kernel_2d<scalar_t><<<num_blocks, threads_per_block>>>(
                        values.data_ptr<scalar_t>(), row_indices.data_ptr<int64_t>(),
                        col_ptrs.data_ptr<int64_t>() + col_start,
                        dense_matrix.data_ptr<scalar_t>() + col_start * dense_col_stride +
                            batch_start * dense_batch_stride,
                        result.data_ptr<scalar_t>() + batch_start * result_batch_stride,
                        cols_in_chunk, batches_in_chunk, dense_col_stride, dense_batch_stride,
                        result_row_stride, result_batch_stride);
                }));
        }
    }

    return result;
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCUDA, m) { m.impl("csc_matmat", &csc_matmat_cuda); }

}  // namespace rlaopt
