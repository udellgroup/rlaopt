#pragma once
#include <ATen/ATen.h>

namespace rlaopt::utils {
// Layout validation
void check_is_sparse_csr(const at::Tensor& tensor, const char* tensor_name = "Input tensor");
void check_is_sparse_csc(const at::Tensor& tensor, const char* tensor_name = "Input tensor");

// Dimension validation
void check_dim(const at::Tensor& tensor, const int64_t expected_dim,
               const char* tensor_name = "Input tensor");
void check_common_dim(const at::Tensor& tensor1, const at::Tensor& tensor2,
                      const char* tensor1_name = "First tensor",
                      const char* tensor2_name = "Second tensor");

// Data type validation
void check_is_floating_point(const at::Tensor& tensor, const char* tensor_name = "Input tensor");
void check_dtype(const at::Tensor& tensor, at::ScalarType expected_dtype,
                 const char* tensor_name = "Input tensor");
void check_same_dtype(const at::Tensor& tensor1, const at::Tensor& tensor2,
                      const char* tensor1_name = "First tensor",
                      const char* tensor2_name = "Second tensor");

// Device validation
void check_is_cpu(const at::Tensor& tensor, const char* tensor_name = "Input tensor");
void check_is_cuda(const at::Tensor& tensor, const char* tensor_name = "Input tensor");
void check_same_device(const at::Tensor& tensor1, const at::Tensor& tensor2,
                       const char* tensor1_name = "First tensor",
                       const char* tensor2_name = "Second tensor");

// Combined input validation
void check_csr_slicing_inputs(const at::Tensor& sparse_tensor, const at::Tensor& row_indices,
                              at::DeviceType device_type, const char* sparse_name = "sparse_tensor",
                              const char* row_name = "row_indices");
void check_csc_matmul_inputs(const at::Tensor& sparse_tensor, const at::Tensor& dense_tensor,
                             at::DeviceType device_type, const int64_t expected_dim,
                             const char* sparse_name = "sparse_tensor",
                             const char* dense_name = "dense_tensor");
}  // namespace rlaopt::utils
