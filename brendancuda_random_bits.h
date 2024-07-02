#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_random_anyrng.h"
#include "brendancuda_binary_basic.h"

namespace BrendanCUDA {
    namespace Random {
        __host__ __device__ uint64_t Get64Bits(uint32_t ProbabilityOf1, AnyRNG<uint64_t> RNG);
    }
}