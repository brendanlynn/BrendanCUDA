#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cstdint"

namespace BrendanCUDA {
    namespace Random {
#ifdef __CUDACC__
        __device__ uint32_t GetSeedOnKernel(uint32_t BaseSeed);
        __device__ uint64_t GetSeedOnKernel(uint64_t BaseSeed);
#endif
        __host__ __device__ uint64_t HashI64(uint64_t Value);
    }
}