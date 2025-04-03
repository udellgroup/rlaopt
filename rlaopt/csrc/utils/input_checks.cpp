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
}  // namespace rlaopt::utils
