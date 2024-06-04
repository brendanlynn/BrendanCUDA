#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cstdint"

namespace BrendanCUDA {
    namespace Random {
        __device__ uint32_t getSeedOnKernel(uint32_t BaseSeed);
        __device__ uint64_t getSeedOnKernel(uint64_t BaseSeed);
        __host__ __device__ uint64_t hashI64(uint64_t v);
    }
}