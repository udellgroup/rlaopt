#include "input_checks.h"

#include <ATen/ATen.h>

namespace rlaopt::utils {
void check_is_sparse_csr(const at::Tensor& tensor, const char* tensor_name) {
    TORCH_CHECK(tensor.layout() == at::kSparseCsr, tensor_name, " must be in CSR format");
}

void check_is_sparse_csc(const at::Tensor& tensor, const char* tensor_name) {
    TORCH_CHECK(tensor.layout() == at::kSparseCsc, tensor_name, " must be in CSC format");
}

void check_dim(const at::Tensor& tensor, const int64_t expected_dim, const char* tensor_name) {
    TORCH_CHECK(tensor.dim() == expected_dim, tensor_name, " must be ", expected_dim,
                "-dimensional");
}

void check_common_dim(const at::Tensor& tensor1, const at::Tensor& tensor2,
                      const char* tensor1_name, const char* tensor2_name) {
    TORCH_CHECK(tensor1.size(1) == tensor2.size(0), "Number of columns in ", tensor1_name,
                " must match number of rows in ", tensor2_name);
}

void check_is_floating_point(const at::Tensor& tensor, const char* tensor_name) {
    TORCH_CHECK(tensor.dtype() == at::kFloat || tensor.dtype() == at::kDouble, tensor_name,
                " must be float32 or float64");
}

void check_dtype(const at::Tensor& tensor, at::ScalarType expected_dtype, const char* tensor_name) {
    TORCH_CHECK(tensor.dtype() == expected_dtype, tensor_name, " must be ",
                toString(expected_dtype), " type");
}

void check_same_dtype(const at::Tensor& tensor1, const at::Tensor& tensor2,
                      const char* tensor1_name, const char* tensor2_name) {
    TORCH_CHECK(tensor1.dtype() == tensor2.dtype(), tensor1_name, " and ", tensor2_name,
                " must have the same dtype");
}

void check_is_cpu(const at::Tensor& tensor, const char* tensor_name) {
    TORCH_CHECK(tensor.device().type() == at::DeviceType::CPU, tensor_name,
                " must be a CPU tensor");
}

void check_is_cuda(const at::Tensor& tensor, const char* tensor_name) {
    TORCH_CHECK(tensor.device().type() == at::DeviceType::CUDA, tensor_name,
                " must be a CUDA tensor");
}

void check_same_device(const at::Tensor& tensor1, const at::Tensor& tensor2,
                       const char* tensor1_name, const char* tensor2_name) {
    TORCH_CHECK(tensor1.device() == tensor2.device(), tensor1_name, " and ", tensor2_name,
                " must be on the same device");
}

void check_csr_slicing_inputs(const at::Tensor& sparse_tensor, const at::Tensor& row_indices,
                              at::DeviceType device_type, const char* sparse_name,
                              const char* row_name) {
    // Common validations
    check_is_sparse_csr(sparse_tensor, sparse_name);
    check_dim(row_indices, 1, row_name);
    check_is_floating_point(sparse_tensor, sparse_name);
    check_dtype(row_indices, at::kLong, row_name);
    check_same_device(sparse_tensor, row_indices, sparse_name, row_name);

    // Device-specific validation
    if (device_type == at::DeviceType::CPU) {
        check_is_cpu(sparse_tensor, sparse_name);
    } else if (device_type == at::DeviceType::CUDA) {
        check_is_cuda(sparse_tensor, sparse_name);
    }
}

void check_csc_matmul_inputs(const at::Tensor& sparse_tensor, const at::Tensor& dense_tensor,
                             at::DeviceType device_type, const int64_t expected_dim,
                             const char* sparse_name, const char* dense_name) {
    // Common validations
    check_is_sparse_csc(sparse_tensor, sparse_name);
    check_dim(dense_tensor, expected_dim, dense_name);
    check_is_floating_point(sparse_tensor, sparse_name);
    check_same_device(sparse_tensor, dense_tensor, sparse_name, dense_name);
    check_same_dtype(sparse_tensor, dense_tensor, sparse_name, dense_name);
    check_common_dim(sparse_tensor, dense_tensor, sparse_name, dense_name);

    // Device-specific validation
    if (device_type == at::DeviceType::CPU) {
        check_is_cpu(sparse_tensor, sparse_name);
    } else if (device_type == at::DeviceType::CUDA) {
        check_is_cuda(sparse_tensor, sparse_name);
    }
}
}  // namespace rlaopt::utils
