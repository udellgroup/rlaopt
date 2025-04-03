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

int get_optimal_block_size(const cudaDeviceProp& props) {
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

    return block_size;
}
}  // namespace rlaopt::cuda_utils
