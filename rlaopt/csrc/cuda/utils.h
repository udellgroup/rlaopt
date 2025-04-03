#pragma once
#include <cuda_runtime.h>

namespace rlaopt::cuda_utils {
// Get properties of the current CUDA device
cudaDeviceProp get_device_properties();

// Helper struct to store device grid limits
struct DeviceGridLimits {
    int max_grid_dim_x;
    int max_grid_dim_y;
};

// Helper to get device maximum grid dimension
DeviceGridLimits get_device_grid_limits(const cudaDeviceProp& props);

// Get optimal thread block configuration for 1D kernels
int get_optimal_block_size_1d(const cudaDeviceProp& props);

// Get optimal thread block configuration for 2D kernels -- useful for CSC matmat
dim3 get_optimal_block_size_2d(const cudaDeviceProp& props, int64_t batch_size);
}  // namespace rlaopt::cuda_utils
