#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

namespace rlaopt {

namespace {
template <typename scalar_t>
void csc_matvec_cpu_impl(const scalar_t* values, const int64_t* row_indices,
                         const int64_t* col_ptrs, const scalar_t* dense_vector_ptr,
                         scalar_t* result_ptr, int64_t num_rows, int64_t num_cols) {
    for (int64_t col = 0; col < num_cols; ++col) {
        scalar_t x_j = dense_vector_ptr[col];

        for (int64_t k = col_ptrs[col]; k < col_ptrs[col + 1]; ++k) {
            int64_t row = row_indices[k];
            scalar_t value = values[k];
            result_ptr[row] += value * x_j;
        }
    }
}
}  // namespace

torch::Tensor csc_matvec_cpu(const torch::Tensor& sparse_tensor,
                             const torch::Tensor& dense_vector) {
    TORCH_CHECK(sparse_tensor.layout() == at::kSparseCsc, "Input tensor must be in CSC format");
    TORCH_CHECK(dense_vector.is_contiguous(), "dense_vector must be contiguous");
    TORCH_CHECK(dense_vector.dim() == 1, "dense_vector must be 1-dimensional");
    TORCH_CHECK(sparse_tensor.device().type() == at::DeviceType::CPU,
                "Input tensor must be on CPU");
    TORCH_CHECK(dense_vector.device().type() == at::DeviceType::CPU, "dense_vector must be on CPU");

    TORCH_CHECK(sparse_tensor.dtype() == dense_vector.dtype(),
                "sparse_tensor and dense_vector must have the same dtype");
    TORCH_CHECK(sparse_tensor.dtype() == torch::kFloat || sparse_tensor.dtype() == torch::kDouble,
                "sparse_tensor must be float32 or float64");

    // Get tensor sizes
    auto num_rows = sparse_tensor.size(0);
    auto num_cols = sparse_tensor.size(1);

    TORCH_CHECK(num_cols == dense_vector.size(0),
                "Number of columns in sparse tensor must match dense vector size");

    // Create result tensor
    auto result = torch::zeros({num_rows}, dense_vector.options());

    // Get row indices and column pointers (same for all data types)
    const int64_t* row_indices = sparse_tensor.row_indices().data_ptr<int64_t>();
    const int64_t* col_ptrs = sparse_tensor.ccol_indices().data_ptr<int64_t>();

    // Use AT_DISPATCH_FLOATING_TYPES to handle different scalar types
    AT_DISPATCH_FLOATING_TYPES(
        sparse_tensor.scalar_type(), "csc_matvec_cpu", ([&] {
            // Get type-specific pointers
            const scalar_t* values = sparse_tensor.values().data_ptr<scalar_t>();
            const scalar_t* dense_vector_ptr = dense_vector.data_ptr<scalar_t>();
            scalar_t* result_ptr = result.data_ptr<scalar_t>();

            // Call implementation with the right type
            csc_matvec_cpu_impl<scalar_t>(values, row_indices, col_ptrs, dense_vector_ptr,
                                          result_ptr, num_rows, num_cols);
        }));

    return result;
}

TORCH_LIBRARY_FRAGMENT(rlaopt, m) {
    m.def("csc_matvec(Tensor sparse_csc_tensor, Tensor dense_vector) -> Tensor");
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCPU, m) { m.impl("csc_matvec", &csc_matvec_cpu); }

}  // namespace rlaopt
