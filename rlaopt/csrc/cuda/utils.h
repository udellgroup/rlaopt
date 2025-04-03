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

// Get optimal thread block configuration
int get_optimal_block_size(const cudaDeviceProp& props);
}  // namespace rlaopt::cuda_utils
