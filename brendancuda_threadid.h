#pragma once

#include "cstdint"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace BrendanCUDA {
#ifdef __CUDACC__
    __device__ __forceinline uint64_t GetThreadID1();
    __device__ __forceinline uint64_t GetThreadID2();
    __device__ __forceinline uint64_t GetThreadID3();
#endif
}

#ifdef __CUDACC__
__device__ __forceinline uint64_t BrendanCUDA::GetThreadID1() {
    uint64_t t = 0;
    t = t * gridDim.x + blockIdx.x;
    t = t * blockDim.x + threadIdx.x;
    return t;
}
__device__ __forceinline uint64_t BrendanCUDA::GetThreadID2() {
    uint64_t t = 0;
    t = t * gridDim.y + blockIdx.y;
    t = t * blockDim.y + threadIdx.y;
    t = t * gridDim.x + blockIdx.x;
    t = t * blockDim.x + threadIdx.x;
    return t;
}
__device__ __forceinline uint64_t BrendanCUDA::GetThreadID3() {
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