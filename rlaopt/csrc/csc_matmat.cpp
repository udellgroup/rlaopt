#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rlaopt {

template <typename scalar_t>
torch::Tensor csc_matmat_cpu_impl(const torch::Tensor& sparse_tensor,
                                  const torch::Tensor& dense_matrix) {
    auto values = sparse_tensor.values();
    auto row_indices = sparse_tensor.row_indices();
    auto col_ptrs = sparse_tensor.ccol_indices();

    auto num_rows = sparse_tensor.size(0);
    auto num_cols = sparse_tensor.size(1);
    auto batch_size = dense_matrix.size(1);  // Number of columns in the dense matrix

    auto result = torch::zeros({num_rows, batch_size}, dense_matrix.options());

    auto values_accessor = values.accessor<scalar_t, 1>();
    auto row_indices_accessor = row_indices.accessor<int64_t, 1>();
    auto col_ptrs_accessor = col_ptrs.accessor<int64_t, 1>();
    auto dense_matrix_accessor = dense_matrix.accessor<scalar_t, 2>();
    auto result_accessor = result.accessor<scalar_t, 2>();

// Parallelize the outer loop (each thread processes a different column of the output)
#pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        // For this column of the dense matrix, compute sparse_matrix * dense_column
        for (int64_t col = 0; col < num_cols; ++col) {
            scalar_t x_jb = dense_matrix_accessor[col][b];

            for (int64_t k = col_ptrs_accessor[col]; k < col_ptrs_accessor[col + 1]; ++k) {
                int64_t row = row_indices_accessor[k];
                scalar_t value = values_accessor[k];
                result_accessor[row][b] += value * x_jb;
            }
        }
    }

    return result;
}

torch::Tensor csc_matmat_cpu(const torch::Tensor& sparse_tensor,
                             const torch::Tensor& dense_matrix) {
    TORCH_CHECK(sparse_tensor.layout() == at::kSparseCsc, "Input tensor must be in CSC format");
    TORCH_CHECK(dense_matrix.is_contiguous(), "dense_matrix must be contiguous");
    TORCH_CHECK(dense_matrix.dim() == 2, "dense_matrix must be 2-dimensional");
    TORCH_CHECK(sparse_tensor.device().type() == at::DeviceType::CPU,
                "Input tensor must be on CPU");
    TORCH_CHECK(dense_matrix.device().type() == at::DeviceType::CPU, "dense_matrix must be on CPU");

    TORCH_CHECK(sparse_tensor.dtype() == dense_matrix.dtype(),
                "sparse_tensor and dense_matrix must have the same dtype");
    TORCH_CHECK(sparse_tensor.dtype() == torch::kFloat || sparse_tensor.dtype() == torch::kDouble,
                "sparse_tensor must be float32 or float64");

    auto num_cols = sparse_tensor.size(1);
    TORCH_CHECK(num_cols == dense_matrix.size(0),
                "Number of columns in sparse tensor must match dense matrix rows");

// Get the number of available threads (for debugging purposes)
// This is useful to check if OpenMP is being used correctly
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    printf("Running CSC matrix-matrix multiplication with %d OpenMP threads\n", num_threads);
#endif

    if (sparse_tensor.dtype() == torch::kFloat) {
        return csc_matmat_cpu_impl<float>(sparse_tensor, dense_matrix);
    } else {
        return csc_matmat_cpu_impl<double>(sparse_tensor, dense_matrix);
    }
}

TORCH_LIBRARY_FRAGMENT(rlaopt, m) {
    m.def("csc_matmat(Tensor sparse_csc_tensor, Tensor dense_matrix) -> Tensor");
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCPU, m) { m.impl("csc_matmat", &csc_matmat_cpu); }

}  // namespace rlaopt
