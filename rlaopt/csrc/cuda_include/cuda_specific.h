#pragma once

namespace rlaopt::utils {
// Struct to store configurations for kernel launches
struct KernelLaunchConfig {
    cudaDeviceProp props;
    int max_grid_dim_x;
    int max_grid_dim_y;
    dim3 block_size;
};

// Get launch configuration for 1D kernels
KernelLaunchConfig get_kernel_launch_config_1d();

// Get launch configuration for 2D kernels
KernelLaunchConfig get_kernel_launch_config_2d(int64_t batch_size);
}  // namespace rlaopt::utils
