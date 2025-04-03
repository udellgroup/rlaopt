#include <cuda_runtime.h>

#include "utils.h"

namespace rlaopt::cuda_utils {
cudaDeviceProp get_device_properties() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props;
}

DeviceGridLimits get_device_grid_limits(const cudaDeviceProp& props) {
    DeviceGridLimits limits;
    limits.max_grid_dim_x = props.maxGridSize[0];
    limits.max_grid_dim_y = props.maxGridSize[1];
    return limits;
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
}  // namespace rlaopt::cuda_utils
