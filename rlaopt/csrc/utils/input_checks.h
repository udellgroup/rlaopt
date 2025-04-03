#pragma once
#include <ATen/ATen.h>

namespace rlaopt::utils {
// Layout validation
void check_is_sparse_csr(const at::Tensor& tensor, const char* tensor_name = "Input tensor");
void check_is_sparse_csc(const at::Tensor& tensor, const char* tensor_name = "Input tensor");

// Dimension validation
void check_dim(const at::Tensor& tensor, const int64_t expected_dim,
               const char* tensor_name = "Input tensor");

// Data type validation
void check_is_floating_point(const at::Tensor& tensor, const char* tensor_name = "Input tensor");
void check_dtype(const at::Tensor& tensor, at::ScalarType expected_dtype,
                 const char* tensor_name = "Input tensor");
void check_same_dtype(const at::Tensor& tensor1, const at::Tensor& tensor2,
                      const char* tensor1_name = "First tensor",
                      const char* tensor2_name = "Second tensor");

// Device validation
void check_is_cuda(const at::Tensor& tensor, const char* tensor_name = "Input tensor");
void check_same_device(const at::Tensor& tensor1, const at::Tensor& tensor2,
                       const char* tensor1_name = "First tensor",
                       const char* tensor2_name = "Second tensor");
}  // namespace rlaopt::utils
