#pragma once

#include "fixedvectors.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace bcuda {
#ifdef __CUDACC__
    __device__ static inline uint32_t GetCoordinates1() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }
    __device__ static inline uint32_2 GetCoordinates2() {
        return uint32_2(
            blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y
        );
    }
    __device__ static inline uint32_3 GetCoordinates3() {
        return uint32_3(
            blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y,
            blockIdx.z * blockDim.z + threadIdx.z
        );
    }
#endif
}