#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace rlaopt {

// CUDA kernel for CSC matrix-matrix product with 2D parallelization and stride handling
template <typename scalar_t>
__global__ void csc_matmat_kernel_2d(
    const scalar_t* __restrict__ values, const int64_t* __restrict__ row_indices,
    const int64_t* __restrict__ col_ptrs, const scalar_t* __restrict__ dense_matrix,
    scalar_t* __restrict__ result, int64_t num_cols, int64_t batch_size, int64_t dense_col_stride,
    int64_t dense_batch_stride, int64_t result_row_stride, int64_t result_batch_stride) {
    // 2D grid: blockIdx.x for dense matrix columns (batch),
    //          blockIdx.y for sparse matrix columns
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && col < num_cols) {
        // Use strides to compute the exact memory location
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

    // Create result tensor with same options as dense_matrix
    auto result = torch::zeros({num_rows, batch_size}, dense_matrix.options());

    // Get strides for proper memory indexing
    auto dense_strides = dense_matrix.strides();
    auto result_strides = result.strides();

    int64_t dense_col_stride = dense_strides[0];  // Stride for moving along columns of dense matrix
    int64_t dense_batch_stride =
        dense_strides[1];  // Stride for moving along batches of dense matrix
    int64_t result_row_stride = result_strides[0];    // Stride for moving along rows of result
    int64_t result_batch_stride = result_strides[1];  // Stride for moving along batches of result

    // Define 2D block and grid dimensions
    dim3 threads_per_block(16, 16);  // 16x16 = 256 threads per block

    // Calculate grid dimensions with proper ceiling division
    dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x,
                    (num_cols + threads_per_block.y - 1) / threads_per_block.y);

    AT_DISPATCH_FLOATING_TYPES(
        sparse_tensor.scalar_type(), "csc_matmat_cuda", ([&] {
            csc_matmat_kernel_2d<scalar_t><<<num_blocks, threads_per_block>>>(
                values.data_ptr<scalar_t>(), row_indices.data_ptr<int64_t>(),
                col_ptrs.data_ptr<int64_t>(), dense_matrix.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(), num_cols, batch_size, dense_col_stride,
                dense_batch_stride, result_row_stride, result_batch_stride);
        }));

    return result;
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCUDA, m) { m.impl("csc_matmat", &csc_matmat_cuda); }

}  // namespace rlaopt
