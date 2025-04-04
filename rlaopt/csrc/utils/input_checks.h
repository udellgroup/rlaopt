#pragma once
#include <ATen/ATen.h>

namespace rlaopt::utils {
// Combined input validation
void check_csr_slicing_inputs(const at::Tensor& sparse_tensor, const at::Tensor& row_indices,
                              at::DeviceType device_type, const char* sparse_name = "sparse_tensor",
                              const char* row_name = "row_indices");
void check_csc_matmul_inputs(const at::Tensor& sparse_tensor, const at::Tensor& dense_tensor,
                             at::DeviceType device_type, const int64_t expected_dim,
                             const char* sparse_name = "sparse_tensor",
                             const char* dense_name = "dense_tensor");
}  // namespace rlaopt::utils
