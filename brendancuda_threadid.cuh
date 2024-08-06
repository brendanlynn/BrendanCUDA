#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cstdint"

namespace BrendanCUDA {
    __device__ __forceinline uint64_t CalculateThreadID1();
    __device__ __forceinline uint64_t CalculateThreadID2();
    __device__ __forceinline uint64_t CalculateThreadID3();
}

__device__ __forceinline uint64_t BrendanCUDA::CalculateThreadID1() {
    uint64_t t = 0;
    t = t * blockDim.x + threadIdx.x;
    t = t * gridDim.x + blockIdx.x;
    return t;
}

__device__ __forceinline uint64_t BrendanCUDA::CalculateThreadID2() {
    uint64_t t = 0;
    t = t * blockDim.x + threadIdx.x;
    t = t * blockDim.y + threadIdx.y;
    t = t * gridDim.x + blockIdx.x;
    t = t * gridDim.y + blockIdx.y;
    return t;
}

__device__ __forceinline uint64_t BrendanCUDA::CalculateThreadID3() {
    uint64_t t = 0;
    t = t * blockDim.x + threadIdx.x;
    t = t * blockDim.y + threadIdx.y;
    t = t * blockDim.z + threadIdx.z;
    t = t * gridDim.x + blockIdx.x;
    t = t * gridDim.y + blockIdx.y;
    t = t * gridDim.z + blockIdx.z;
    return t;
}