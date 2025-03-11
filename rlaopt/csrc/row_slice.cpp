#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

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

at::Tensor get_row_slice_cpu(const at::Tensor& sparse_tensor, const at::Tensor& row_indices) {
    TORCH_CHECK(sparse_tensor.layout() == at::kSparseCsr, "Input tensor must be in CSR format");
    TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1-dimensional");
    TORCH_CHECK(sparse_tensor.device().type() == at::DeviceType::CPU,
                "Input tensor must be on CPU");
    TORCH_CHECK(row_indices.device().type() == at::DeviceType::CPU, "row_indices must be on CPU");

    auto values = sparse_tensor.values();
    auto crow_indices = sparse_tensor.crow_indices();
    auto col_indices = sparse_tensor.col_indices();

    auto num_requested_rows = row_indices.size(0);

    // Calculate the total number of non-zero elements in the selected rows
    auto new_nnz = at::zeros({1}, crow_indices.options());
    for (int64_t i = 0; i < num_requested_rows; i++) {
        auto row = row_indices.index({i}).item<int64_t>();
        new_nnz += crow_indices.index({row + 1}) - crow_indices.index({row});
    }

    // Create new tensors for the result
    auto new_values = at::empty(new_nnz.item<int64_t>(), values.options());
    auto new_col_indices = at::empty(new_nnz.item<int64_t>(), col_indices.options());
    auto new_crow_indices = at::empty(num_requested_rows + 1, crow_indices.options());

    int64_t current_nnz = 0;
    new_crow_indices.index_put_({0}, 0);

    for (int64_t i = 0; i < num_requested_rows; i++) {
        auto row = row_indices.index({i}).item<int64_t>();
        auto start = crow_indices.index({row}).item<int64_t>();
        auto end = crow_indices.index({row + 1}).item<int64_t>();
        auto row_nnz = end - start;

        new_values.index_copy_(0, at::arange(current_nnz, current_nnz + row_nnz),
                               values.slice(0, start, end));
        new_col_indices.index_copy_(0, at::arange(current_nnz, current_nnz + row_nnz),
                                    col_indices.slice(0, start, end));

        current_nnz += row_nnz;
        new_crow_indices.index_put_({i + 1}, current_nnz);
    }

    // Create const references to the tensors
    const at::Tensor& const_new_crow_indices = new_crow_indices;
    const at::Tensor& const_new_col_indices = new_col_indices;
    const at::Tensor& const_new_values = new_values;

    return at::sparse_csr_tensor(const_new_crow_indices, const_new_col_indices, const_new_values,
                                 {num_requested_rows, sparse_tensor.size(1)},
                                 sparse_tensor.options());
}

TORCH_LIBRARY_FRAGMENT(rlaopt, m) {
    m.def("get_row_slice(Tensor sparse_csr_tensor, Tensor row_indices) -> Tensor");
}

// Registers SparseCsrCUDA backend for get_row_slice
TORCH_LIBRARY_IMPL(rlaopt, SparseCsrCPU, m) { m.impl("get_row_slice", &get_row_slice_cpu); }

}  // namespace rlaopt
