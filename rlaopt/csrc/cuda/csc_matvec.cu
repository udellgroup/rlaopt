#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace rlaopt {

// CUDA kernel for CSC matrix-vector product
template <typename scalar_t>
__global__ void csc_matvec_kernel(const scalar_t* __restrict__ values,
                                  const int64_t* __restrict__ row_indices,
                                  const int64_t* __restrict__ col_ptrs,
                                  const scalar_t* __restrict__ dense_vector,
                                  scalar_t* __restrict__ result, int64_t num_cols) {
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

    const int threads_per_block = 256;
    const int num_blocks = (num_cols + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(sparse_tensor.scalar_type(), "csc_matvec_cuda", ([&] {
                                   csc_matvec_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                                       values.data_ptr<scalar_t>(), row_indices.data_ptr<int64_t>(),
                                       col_ptrs.data_ptr<int64_t>(),
                                       dense_vector.data_ptr<scalar_t>(),
                                       result.data_ptr<scalar_t>(), num_cols);
                               }));

    return result;
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCUDA, m) { m.impl("csc_matvec", &csc_matvec_cuda); }

}  // namespace rlaopt
