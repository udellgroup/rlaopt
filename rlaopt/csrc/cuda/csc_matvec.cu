#include <ATen/Operators.h>
#include <torch/library.h>

#include "../cpp_include/input_checks.h"
#include "../cuda_include/cuda_specific.h"

namespace rlaopt {

namespace {
// CUDA kernel for CSC matrix-vector product
template <typename scalar_t>
__global__ void csc_matvec_kernel(const scalar_t* __restrict__ values,
                                  const int64_t* __restrict__ row_indices,
                                  const int64_t* __restrict__ col_ptrs,
                                  const scalar_t* __restrict__ dense_tensor,
                                  scalar_t* __restrict__ result, const int64_t num_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_cols) {
        scalar_t x_j = dense_tensor[col];

        for (int64_t k = col_ptrs[col]; k < col_ptrs[col + 1]; ++k) {
            int64_t row = row_indices[k];
            scalar_t value = values[k];
            atomicAdd(&result[row], value * x_j);
        }
    }
}
}  // namespace

at::Tensor csc_matvec_cuda(const at::Tensor& sparse_tensor, const at::Tensor& dense_tensor) {
    rlaopt::utils::check_csc_matmul_inputs(sparse_tensor, dense_tensor, at::DeviceType::CUDA, 1,
                                           "sparse_tensor", "dense_tensor");

    auto values = sparse_tensor.values();
    auto row_indices = sparse_tensor.row_indices();
    auto col_ptrs = sparse_tensor.ccol_indices();

    auto num_rows = sparse_tensor.size(0);
    auto num_cols = sparse_tensor.size(1);

    TORCH_CHECK(num_cols == dense_tensor.size(0),
                "Number of columns in sparse tensor must match dense vector size");

    auto result = at::zeros({num_rows}, dense_tensor.options());

    // Get kernel launch configuration
    rlaopt::utils::KernelLaunchConfig config = rlaopt::utils::get_kernel_launch_config_1d();
    dim3 threads_per_block = config.block_size;
    int64_t MAX_GRID_DIM_X = config.max_grid_dim_x;

    // Process the matrix in column chunks if needed
    for (int64_t col_start = 0; col_start < num_cols;
         col_start += MAX_GRID_DIM_X * threads_per_block.x) {
        int64_t cols_in_chunk =
            std::min(MAX_GRID_DIM_X * threads_per_block.x, num_cols - col_start);
        int num_blocks = (cols_in_chunk + threads_per_block.x - 1) / threads_per_block.x;

        // // Uncomment for debugging
        // printf("Processing columns %ld to %ld (Blocks: %d, Threads per block: %d)\n",
        //        col_start, col_start + cols_in_chunk - 1, num_blocks, threads_per_block.x);

        AT_DISPATCH_FLOATING_TYPES(
            sparse_tensor.scalar_type(), "csc_matvec_cuda", ([&] {
                csc_matvec_kernel<scalar_t><<<num_blocks, threads_per_block.x>>>(
                    values.data_ptr<scalar_t>(), row_indices.data_ptr<int64_t>(),
                    col_ptrs.data_ptr<int64_t>() + col_start,
                    dense_tensor.data_ptr<scalar_t>() + col_start, result.data_ptr<scalar_t>(),
                    cols_in_chunk);
            }));
    }

    return result;
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCUDA, m) { m.impl("csc_matvec", &csc_matvec_cuda); }

}  // namespace rlaopt
