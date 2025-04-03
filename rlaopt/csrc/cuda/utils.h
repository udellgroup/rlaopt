#pragma once
#include <cuda_runtime.h>

namespace rlaopt::cuda_utils {
// Get properties of the current CUDA device
void get_device_properties(cudaDeviceProp& props);

// Get optimal thread block configuration for 1D kernels
dim3 get_optimal_block_size_1d(const cudaDeviceProp& props);

// Get optimal thread block configuration for 2D kernels -- useful for CSC matmat
dim3 get_optimal_block_size_2d(const cudaDeviceProp& props, int64_t batch_size);

// Struct to store configurations for kernel launches
struct KernelLaunchConfig {
    cudaDeviceProp props;
    int max_grid_dim_x;
    int max_grid_dim_y;
    dim3 block_size;
};

// Get base launch configuration
void get_base_kernel_launch_config(KernelLaunchConfig& config);

// Get launch configuration for 1D kernels
KernelLaunchConfig get_kernel_launch_config_1d();

// Get launch configuration for 2D kernels
KernelLaunchConfig get_kernel_launch_config_2d(int64_t batch_size);
}  // namespace rlaopt::cuda_utils
