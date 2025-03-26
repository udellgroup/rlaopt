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
    // Use 2D thread blocks and 2D grid for maximum parallelism
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

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

    // 2D thread blocks - optimize for both dimensions
    // We want a total of 256 threads per block (good occupancy)
    dim3 threads_per_block(16, 16);  // 16x16 = 256 threads per block
    const int64_t MAX_GRID_DIM = 65535;

    // Process the matrix in column and batch chunks to stay within CUDA limits
    for (int64_t col_start = 0; col_start < num_cols;
         col_start += MAX_GRID_DIM * threads_per_block.y) {
        int64_t cols_in_chunk = std::min(MAX_GRID_DIM * threads_per_block.y, num_cols - col_start);
        int grid_y = (cols_in_chunk + threads_per_block.y - 1) / threads_per_block.y;

        for (int64_t batch_start = 0; batch_start < batch_size;
             batch_start += MAX_GRID_DIM * threads_per_block.x) {
            int64_t batches_in_chunk =
                std::min(MAX_GRID_DIM * threads_per_block.x, batch_size - batch_start);
            int grid_x = (batches_in_chunk + threads_per_block.x - 1) / threads_per_block.x;

            dim3 num_blocks(grid_x, grid_y);

            // printf("Processing columns %lld to %lld, batches %lld to %lld (Grid: %d×%d, Block:
            // 16×16)\n",
            //       col_start, col_start + cols_in_chunk - 1,
            //       batch_start, batch_start + batches_in_chunk - 1,
            //       grid_x, grid_y);

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
