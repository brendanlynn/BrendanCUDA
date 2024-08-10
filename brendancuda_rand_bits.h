#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_rand_anyrng.h"
#include "brendancuda_binary_basic.h"

namespace BrendanCUDA {
    namespace Random {
        __host__ __device__ __forceinline uint64_t Get64Bits(uint32_t ProbabilityOf1, AnyRNG<uint64_t> RNG);
    }
}


__host__ __device__ __forceinline uint64_t BrendanCUDA::Random::Get64Bits(uint32_t ProbabilityOf1, AnyRNG<uint64_t> RNG) {
    uint32_t ct = BrendanCUDA::Binary::CountBitsB(ProbabilityOf1);
    if (!ct) {
        return 0;
    }
    uint32_t lb = 1u << 31 >> (ct - 1);
    uint64_t cr = RNG();
    for (uint32_t i = 1ui64 << 31; i > lb; i >>= 1) {
        if (ProbabilityOf1 & i) {
            cr |= RNG();
        }
        else {
            cr &= RNG();
        }
    }
    return cr;
}