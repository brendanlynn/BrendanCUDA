#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_random_anyrng.cuh"
#include "brendancuda_binary_basic.cuh"

namespace BrendanCUDA {
    namespace Random {
        __host__ __device__ uint64_t Get64Bits(uint32_t ProbabilityOf1, AnyRNG<uint64_t> RNG);
    }
}