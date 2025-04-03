#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_specific.h"

namespace rlaopt {

namespace {
__global__ void compute_row_nnz_kernel(const int64_t num_requested_rows,
                                       const int64_t* crow_indices, const int64_t* row_indices,
                                       int64_t* new_crow_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_requested_rows) {
        int64_t row = row_indices[idx];
        new_crow_indices[idx] = crow_indices[row + 1] - crow_indices[row];
    }
}

template <typename scalar_t>
__global__ void copy_values_and_indices_kernel(const int64_t num_requested_rows,
                                               const int64_t* crow_indices,
                                               const int64_t* col_indices, const scalar_t* values,
                                               const int64_t* row_indices,
                                               const int64_t* new_crow_indices,
                                               int64_t* new_col_indices, scalar_t* new_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_requested_rows) {
        int64_t row = row_indices[idx];
        int64_t start = crow_indices[row];
        int64_t end = crow_indices[row + 1];
        int64_t new_start = new_crow_indices[idx];

        for (int64_t i = start; i < end; ++i) {
            int64_t new_idx = new_start + (i - start);
            new_col_indices[new_idx] = col_indices[i];
            new_values[new_idx] = values[i];
        }
    }
}
}  // namespace

at::Tensor get_row_slice_cuda(const at::Tensor& sparse_tensor, const at::Tensor& row_indices) {
    TORCH_CHECK(sparse_tensor.layout() == at::kSparseCsr, "Input tensor must be in CSR format");
    TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1-dimensional");
    TORCH_CHECK(sparse_tensor.dtype() == at::kFloat || sparse_tensor.dtype() == at::kDouble,
                "Input tensor must be float32 or float64");
    TORCH_CHECK(row_indices.dtype() == at::kLong, "Row indices must be long");
    TORCH_CHECK(sparse_tensor.device().type() == at::DeviceType::CUDA,
                "Sparse tensor must be a CUDA tensor");
    TORCH_CHECK(row_indices.device().type() == at::DeviceType::CUDA,
                "Row indices must be a CUDA tensor");
    TORCH_CHECK(sparse_tensor.device() == row_indices.device(),
                "Sparse tensor and row indices must be on the same CUDA device");

    auto values = sparse_tensor.values();
    auto crow_indices = sparse_tensor.crow_indices();
    auto col_indices = sparse_tensor.col_indices();

    const int64_t num_requested_rows = row_indices.size(0);

    // Compute new crow_indices
    auto new_crow_indices = at::zeros({num_requested_rows + 1}, crow_indices.options());

    // Get kernel launch configuration
    rlaopt::cuda_utils::KernelLaunchConfig config =
        rlaopt::cuda_utils::get_kernel_launch_config_1d();
    dim3 threads_per_block = config.block_size;
    int64_t MAX_GRID_DIM_X = config.max_grid_dim_x;

    // Process the matrix in row chunks if needed
    for (int64_t row_start = 0; row_start < num_requested_rows;
         row_start += MAX_GRID_DIM_X * threads_per_block.x) {
        int64_t rows_in_chunk =
            std::min(MAX_GRID_DIM_X * threads_per_block.x, num_requested_rows - row_start);
        int num_blocks = (rows_in_chunk + threads_per_block.x - 1) / threads_per_block.x;

        // Compute number of non-zero elements in each row
        compute_row_nnz_kernel<<<num_blocks, threads_per_block.x>>>(
            rows_in_chunk, crow_indices.data_ptr<int64_t>(),
            row_indices.data_ptr<int64_t>() + row_start,
            new_crow_indices.data_ptr<int64_t>() + 1 +
                row_start  // +1 because we start filling from index 1
        );
    }

    // Compute prefix sum to get final crow_indices
    new_crow_indices = at::cumsum(new_crow_indices, 0);

    // Get total number of non-zero elements
    int64_t new_nnz = new_crow_indices[-1].item<int64_t>();

    // Allocate memory for new tensors
    auto new_values = at::empty(new_nnz, values.options());
    auto new_col_indices = at::empty(new_nnz, col_indices.options());

    // Process the matrix in row chunks if needed
    for (int64_t row_start = 0; row_start < num_requested_rows;
         row_start += MAX_GRID_DIM_X * threads_per_block.x) {
        int64_t rows_in_chunk =
            std::min(MAX_GRID_DIM_X * threads_per_block.x, num_requested_rows - row_start);
        int num_blocks = (rows_in_chunk + threads_per_block.x - 1) / threads_per_block.x;

        // Copy values and indices
        AT_DISPATCH_FLOATING_TYPES(
            sparse_tensor.scalar_type(), "get_row_slice_cuda", ([&] {
                copy_values_and_indices_kernel<scalar_t><<<num_blocks, threads_per_block.x>>>(
                    rows_in_chunk, crow_indices.data_ptr<int64_t>(),
                    col_indices.data_ptr<int64_t>(), values.data_ptr<scalar_t>(),
                    row_indices.data_ptr<int64_t>() + row_start,
                    new_crow_indices.data_ptr<int64_t>() + row_start,
                    new_col_indices.data_ptr<int64_t>(), new_values.data_ptr<scalar_t>());
            }));
    }

    return at::sparse_csr_tensor(new_crow_indices, new_col_indices, new_values,
                                 {num_requested_rows, sparse_tensor.size(1)},
                                 sparse_tensor.options());
}

// Registers SparseCsrCUDA backend for get_row_slice
TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCUDA, m) { m.impl("get_row_slice", &get_row_slice_cuda); }

}  // namespace rlaopt
