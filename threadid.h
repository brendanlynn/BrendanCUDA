#pragma once

#include "cstdint"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace bcuda {
#ifdef __CUDACC__
    __device__ static inline uint64_t GetThreadID1() {
        uint64_t t = 0;
        t = t * gridDim.x + blockIdx.x;
        t = t * blockDim.x + threadIdx.x;
        return t;
    }
    __device__ static inline uint64_t GetThreadID2() {
        uint64_t t = 0;
        t = t * gridDim.y + blockIdx.y;
        t = t * blockDim.y + threadIdx.y;
        t = t * gridDim.x + blockIdx.x;
        t = t * blockDim.x + threadIdx.x;
        return t;
    }
    __device__ static inline uint64_t GetThreadID3() {
        uint64_t t = 0;
        t = t * gridDim.z + blockIdx.z;
        t = t * blockDim.z + threadIdx.z;
        t = t * gridDim.y + blockIdx.y;
        t = t * blockDim.y + threadIdx.y;
        t = t * gridDim.x + blockIdx.x;
        t = t * blockDim.x + threadIdx.x;
        return t;
    }
#endif
}