#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../utils/input_checks.h"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}
}

namespace rlaopt {

namespace {
template <typename scalar_t>
void get_row_slice_cpu_impl(const scalar_t* values_ptr, const int64_t* crow_indices_ptr,
                            const int64_t* col_indices_ptr, const int64_t* row_indices_ptr,
                            scalar_t* new_values_ptr, int64_t* new_col_indices_ptr,
                            int64_t* new_crow_indices_ptr, int64_t num_requested_rows,
                            int64_t num_cols) {
    int64_t current_nnz = 0;
    new_crow_indices_ptr[0] = 0;

    for (int64_t i = 0; i < num_requested_rows; i++) {
        int64_t row = row_indices_ptr[i];
        int64_t start = crow_indices_ptr[row];
        int64_t end = crow_indices_ptr[row + 1];
        int64_t row_nnz = end - start;

        // Copy values and column indices for this row
        for (int64_t j = 0; j < row_nnz; j++) {
            new_values_ptr[current_nnz + j] = values_ptr[start + j];
            new_col_indices_ptr[current_nnz + j] = col_indices_ptr[start + j];
        }

        current_nnz += row_nnz;
        new_crow_indices_ptr[i + 1] = current_nnz;
    }
}
}  // namespace

at::Tensor get_row_slice_cpu(const at::Tensor& sparse_tensor, const at::Tensor& row_indices) {
    rlaopt::utils::check_is_sparse_csr(sparse_tensor, "sparse_tensor");
    rlaopt::utils::check_dim(row_indices, 1, "row_indices");
    rlaopt::utils::check_is_floating_point(sparse_tensor, "sparse_tensor");
    rlaopt::utils::check_dtype(row_indices, at::kLong, "row_indices");
    rlaopt::utils::check_same_device(sparse_tensor, row_indices, "sparse_tensor", "row_indices");
    rlaopt::utils::check_is_cpu(sparse_tensor, "sparse_tensor");

    // Get sizes and pointers
    auto num_requested_rows = row_indices.size(0);
    auto num_cols = sparse_tensor.size(1);
    auto row_indices_ptr = row_indices.data_ptr<int64_t>();

    // Get CSR components
    auto values = sparse_tensor.values();
    auto crow_indices = sparse_tensor.crow_indices();
    auto col_indices = sparse_tensor.col_indices();

    // Get pointers for CSR components
    auto crow_indices_ptr = crow_indices.data_ptr<int64_t>();
    auto col_indices_ptr = col_indices.data_ptr<int64_t>();

    // Calculate the total number of non-zero elements in the selected rows
    int64_t new_nnz = 0;
    for (int64_t i = 0; i < num_requested_rows; i++) {
        int64_t row = row_indices_ptr[i];
        new_nnz += crow_indices_ptr[row + 1] - crow_indices_ptr[row];
    }

    // Create new tensors for the result
    auto new_crow_indices = torch::empty({num_requested_rows + 1}, crow_indices.options());
    auto new_col_indices = torch::empty({new_nnz}, col_indices.options());
    auto new_values = torch::empty({new_nnz}, values.options());

    // Get pointers for new tensors
    auto new_crow_indices_ptr = new_crow_indices.data_ptr<int64_t>();
    auto new_col_indices_ptr = new_col_indices.data_ptr<int64_t>();

    // Use type dispatch to handle float and double
    AT_DISPATCH_FLOATING_TYPES(sparse_tensor.scalar_type(), "get_row_slice_cpu", ([&] {
                                   // Get type-specific pointers
                                   const scalar_t* values_ptr = values.data_ptr<scalar_t>();
                                   scalar_t* new_values_ptr = new_values.data_ptr<scalar_t>();

                                   // Call implementation
                                   get_row_slice_cpu_impl<scalar_t>(
                                       values_ptr, crow_indices_ptr, col_indices_ptr,
                                       row_indices_ptr, new_values_ptr, new_col_indices_ptr,
                                       new_crow_indices_ptr, num_requested_rows, num_cols);
                               }));

    // Create the result sparse tensor
    return at::sparse_csr_tensor(new_crow_indices, new_col_indices, new_values,
                                 {num_requested_rows, num_cols}, sparse_tensor.options());
}

TORCH_LIBRARY_FRAGMENT(rlaopt, m) {
    m.def("get_row_slice(Tensor sparse_csr_tensor, Tensor row_indices) -> Tensor");
}

// Registers SparseCsrCPU backend for get_row_slice
TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCPU, m) { m.impl("get_row_slice", &get_row_slice_cpu); }

}  // namespace rlaopt
