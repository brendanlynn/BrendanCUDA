#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_rand_anyrng.h"
#include "brendancuda_binary_basic.h"

namespace BrendanCUDA {
    namespace Random {
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __device__ __forceinline uint64_t Get64Bits(uint32_t ProbabilityOf1, _TRNG RNG);
    }
}

template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Random::Get64Bits(uint32_t ProbabilityOf1, _TRNG RNG) {
    uint32_t ct = BrendanCUDA::Binary::CountBitsB(ProbabilityOf1);
    if (!ct) {
        return 0;
    }
    std::uniform_int_distribution<uint64_t> dis64(0);
    uint32_t lb = 1u << 31 >> (ct - 1);
    uint64_t cr = dis64(RNG);
    for (uint32_t i = 1ui64 << 31; i > lb; i >>= 1) {
        if (ProbabilityOf1 & i) {
            cr |= dis64(RNG);
        }
        else {
            cr &= dis64(RNG);
        }
    }
    return cr;
}