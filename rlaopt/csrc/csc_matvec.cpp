#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

namespace rlaopt {

template <typename scalar_t>
torch::Tensor csc_matvec_cpu_impl(const torch::Tensor& sparse_tensor,
                                  const torch::Tensor& dense_vector) {
    auto values = sparse_tensor.values();
    auto row_indices = sparse_tensor.row_indices();
    auto col_ptrs = sparse_tensor.ccol_indices();

    auto num_rows = sparse_tensor.size(0);
    auto num_cols = sparse_tensor.size(1);

    auto result = torch::zeros({num_rows}, dense_vector.options());

    auto values_accessor = values.accessor<scalar_t, 1>();
    auto row_indices_accessor = row_indices.accessor<int64_t, 1>();
    auto col_ptrs_accessor = col_ptrs.accessor<int64_t, 1>();
    auto dense_vector_accessor = dense_vector.accessor<scalar_t, 1>();
    auto result_accessor = result.accessor<scalar_t, 1>();

    for (int64_t col = 0; col < num_cols; ++col) {
        scalar_t x_j = dense_vector_accessor[col];

        // Skip if the entry in the dense matrix is zero (optimization)
        if (x_j == 0) continue;

        for (int64_t k = col_ptrs_accessor[col]; k < col_ptrs_accessor[col + 1]; ++k) {
            int64_t row = row_indices_accessor[k];
            scalar_t value = values_accessor[k];
            result_accessor[row] += value * x_j;
        }
    }

    return result;
}

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

    auto num_cols = sparse_tensor.size(1);
    TORCH_CHECK(num_cols == dense_vector.size(0),
                "Number of columns in sparse tensor must match dense vector size");

    if (sparse_tensor.dtype() == torch::kFloat) {
        return csc_matvec_cpu_impl<float>(sparse_tensor, dense_vector);
    } else {
        return csc_matvec_cpu_impl<double>(sparse_tensor, dense_vector);
    }
}

TORCH_LIBRARY_FRAGMENT(rlaopt, m) {
    m.def("csc_matvec(Tensor sparse_csc_tensor, Tensor dense_vector) -> Tensor");
}

TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCPU, m) { m.impl("csc_matvec", &csc_matvec_cpu); }

}  // namespace rlaopt
