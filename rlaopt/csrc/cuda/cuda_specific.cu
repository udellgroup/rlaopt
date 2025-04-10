#include <cuda_runtime.h>

#include "../cuda_include/cuda_specific.h"

namespace rlaopt::utils {

namespace {
// Internal helper functions
void get_device_properties(cudaDeviceProp& props) {
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
}

void get_base_kernel_launch_config(KernelLaunchConfig& config) {
    get_device_properties(config.props);
    config.max_grid_dim_x = config.props.maxGridSize[0];
    config.max_grid_dim_y = config.props.maxGridSize[1];
}

dim3 get_optimal_block_size_1d(const cudaDeviceProp& props) {
    int max_threads_per_block = props.maxThreadsPerBlock;
    int warp_size = props.warpSize;

    // Default to 256 threads per block for good occupancy
    int block_size = 256;

    // Adjust based on device capabilities
    if (max_threads_per_block < 1024) {
        block_size = max_threads_per_block / 2;
    }

    // Ensure block_size is a multiple of warp_size
    block_size = (block_size / warp_size) * warp_size;

    return dim3(block_size);
}

dim3 get_optimal_block_size_2d(const cudaDeviceProp& props, int64_t batch_size) {
    // Calculate target threads based on device capabilities
    // Most devices work well with 256 threads per block, but we can be more precise
    int max_threads_per_block = props.maxThreadsPerBlock;
    int warp_size = props.warpSize;

    // Best practice: Use a multiple of 32 (warp size) for total thread count
    // Aim for 25-75% of max threads per block for good occupancy
    int target_threads = 256;  // Default starting point

    // Adjust based on device capabilities
    if (max_threads_per_block < 1024) {
        // For older devices or those with smaller limits
        target_threads = max_threads_per_block / 2;
    } else {
        // For modern devices, use a value that achieves good occupancy
        // typically 256 or 512 works well
        target_threads = 256;
    }

    // Ensure target_threads is a multiple of warp_size
    target_threads = (target_threads / warp_size) * warp_size;

    // Start with default configuration
    int threads_x = 16;                   // For columns (just 1 column per thread)
    int threads_y = target_threads / 16;  // For batches (process many batches per block)

    // If batch_size is too small to fill a block
    if (batch_size < target_threads / 16) {
        if (batch_size <= target_threads / 32) {
            // Very small batch size - distribute threads to process more columns
            threads_y = batch_size;
            threads_x = target_threads / threads_y;
            // Ensure threads_x is a power of 2 for better memory alignment
            threads_x = 1 << (int)log2(threads_x);
        } else {
            // Moderate batch size - just use the batch_size
            threads_y = batch_size;
        }
    }

    return dim3(threads_x, threads_y);
}
}  // namespace

KernelLaunchConfig get_kernel_launch_config_1d() {
    KernelLaunchConfig config;
    get_base_kernel_launch_config(config);
    config.block_size = get_optimal_block_size_1d(config.props);
    return config;
}

KernelLaunchConfig get_kernel_launch_config_2d(int64_t batch_size) {
    KernelLaunchConfig config;
    get_base_kernel_launch_config(config);
    config.block_size = get_optimal_block_size_2d(config.props, batch_size);
    return config;
}
}  // namespace rlaopt::utils
