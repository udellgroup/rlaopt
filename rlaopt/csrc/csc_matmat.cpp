#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rlaopt {

template <typename scalar_t>
void csc_matmat_cpu_impl(const scalar_t* values, const int64_t* row_indices,
                         const int64_t* col_ptrs, const scalar_t* dense_matrix_ptr,
                         scalar_t* result_ptr, int64_t num_rows, int64_t num_cols,
                         int64_t batch_size, int64_t dense_col_stride, int64_t dense_batch_stride,
                         int64_t result_row_stride, int64_t result_batch_stride) {
// Parallelize the outer loop (each thread processes a different column of the output)
#pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        // For this column of the dense matrix, compute sparse_matrix * dense_column
        for (int64_t col = 0; col < num_cols; ++col) {
            scalar_t x_jb = dense_matrix_ptr[col * dense_col_stride + b * dense_batch_stride];

            // Skip computation if value is zero
            if (x_jb == 0) continue;

            for (int64_t k = col_ptrs[col]; k < col_ptrs[col + 1]; ++k) {
                int64_t row = row_indices[k];
                scalar_t value = values[k];
                result_ptr[row * result_row_stride + b * result_batch_stride] += value * x_jb;
            }
        }
    }
}

torch::Tensor csc_matmat_cpu(const torch::Tensor& sparse_tensor,
                             const torch::Tensor& dense_matrix) {
    TORCH_CHECK(sparse_tensor.layout() == at::kSparseCsc, "Input tensor must be in CSC format");
    TORCH_CHECK(dense_matrix.dim() == 2, "dense_matrix must be 2-dimensional");
    TORCH_CHECK(sparse_tensor.device().type() == at::DeviceType::CPU,
                "Input tensor must be on CPU");
    TORCH_CHECK(dense_matrix.device().type() == at::DeviceType::CPU, "dense_matrix must be on CPU");

    TORCH_CHECK(sparse_tensor.dtype() == dense_matrix.dtype(),
                "sparse_tensor and dense_matrix must have the same dtype");
    TORCH_CHECK(sparse_tensor.dtype() == torch::kFloat || sparse_tensor.dtype() == torch::kDouble,
                "sparse_tensor must be float32 or float64");

    // Get tensor sizes
    auto num_rows = sparse_tensor.size(0);
    auto num_cols = sparse_tensor.size(1);
    auto batch_size = dense_matrix.size(1);

    TORCH_CHECK(num_cols == dense_matrix.size(0),
                "Number of columns in sparse tensor must match dense matrix rows");

    // Get strides for efficient memory access
    auto dense_strides = dense_matrix.strides();
    int64_t dense_col_stride = dense_strides[0];
    int64_t dense_batch_stride = dense_strides[1];

    // Create result tensor
    auto result = torch::zeros({num_rows, batch_size}, dense_matrix.options());
    auto result_strides = result.strides();
    int64_t result_row_stride = result_strides[0];
    int64_t result_batch_stride = result_strides[1];

// Get the number of available threads (for debugging purposes)
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    printf("Running CSC matrix-matrix multiplication with %d OpenMP threads\n", num_threads);
#endif

    // Get row indices and column pointers (same for all data types)
    const int64_t* row_indices = sparse_tensor.row_indices().data_ptr<int64_t>();
    const int64_t* col_ptrs = sparse_tensor.ccol_indices().data_ptr<int64_t>();

    // Use AT_DISPATCH_FLOATING_TYPES to handle different scalar types
    AT_DISPATCH_FLOATING_TYPES(
        sparse_tensor.scalar_type(), "csc_matmat_cpu", ([&] {
            // Get type-specific pointers
            const scalar_t* values = sparse_tensor.values().data_ptr<scalar_t>();
            const scalar_t* dense_matrix_ptr = dense_matrix.data_ptr<scalar_t>();
            scalar_t* result_ptr = result.data_ptr<scalar_t>();

            // Call implementation with the right type
            csc_matmat_cpu_impl<scalar_t>(values, row_indices, col_ptrs, dense_matrix_ptr,
                                          result_ptr, num_rows, num_cols, batch_size,
                                          dense_col_stride, dense_batch_stride, result_row_stride,
                                          result_batch_stride);
        }));

    return result;
}

TORCH_LIBRARY_FRAGMENT(rlaopt, m) {
    m.def("csc_matmat(Tensor sparse_csc_tensor, Tensor dense_matrix) -> Tensor");
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCPU, m) { m.impl("csc_matmat", &csc_matmat_cpu); }

}  // namespace rlaopt
