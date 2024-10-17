#pragma once

#include "binary_basic.h"
#include "curandkernelgens.h"
#include "rand_anyrng.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace bcuda {
    namespace rand {
        template <std::uniform_random_bit_generator _TRNG>
        __host__ inline uint32_t Get32Bits(uint32_t ProbabilityOf1, _TRNG& RNG) {
            uint32_t ct = bcuda::binary::CountBitsB(ProbabilityOf1);
            if (!ct) return 0;
            std::uniform_int_distribution<uint32_t> dis32(0);
            uint32_t lb = 1u << 31 >> (ct - 1);
            uint32_t cr = dis32(RNG);
            for (uint32_t i = 1ui64 << 31; i > lb; i >>= 1) {
                if (ProbabilityOf1 & i)
                    cr |= dis32(RNG);
                else
                    cr &= dis32(RNG);
            }
            return cr;
        }
        template <std::uniform_random_bit_generator _TRNG>
        __host__ inline uint64_t Get64Bits(uint32_t ProbabilityOf1, _TRNG& RNG) {
            uint32_t ct = bcuda::binary::CountBitsB(ProbabilityOf1);
            if (!ct) return 0;
            std::uniform_int_distribution<uint64_t> dis64(0);
            uint32_t lb = 1u << 31 >> (ct - 1);
            uint64_t cr = dis64(RNG);
            for (uint32_t i = 1ui64 << 31; i > lb; i >>= 1) {
                if (ProbabilityOf1 & i)
                    cr |= dis64(RNG);
                else
                    cr &= dis64(RNG);
            }
            return cr;
        }
#ifdef __CUDACC__
        template <KernelCurandState _TRNG>
        __device__ inline uint32_t Get32Bits(uint32_t ProbabilityOf1, _TRNG& RNG) {
            uint32_t ct = bcuda::binary::CountBitsB(ProbabilityOf1);
            if (!ct) return 0;
            uint32_t lb = 1u << 31 >> (ct - 1);
            uint32_t cr = curand(&RNG);
            for (uint32_t i = 1ui64 << 31; i > lb; i >>= 1) {
                if (ProbabilityOf1 & i)
                    cr |= curand(&RNG);
                else
                    cr &= curand(&RNG);
            }
            return cr;
        }
        template <KernelCurandState _TRNG>
        __device__ inline uint64_t Get64Bits(uint32_t ProbabilityOf1, _TRNG& RNG) {
            uint32_t ct = bcuda::binary::CountBitsB(ProbabilityOf1);
            if (!ct) return 0;
            uint32_t lb = 1u << 31 >> (ct - 1);
            uint64_t cr = ((uint64_t)curand(&RNG) << 32) | curand(&RNG);
            for (uint32_t i = 1ui64 << 31; i > lb; i >>= 1) {
                if (ProbabilityOf1 & i)
                    cr |= ((uint64_t)curand(&RNG) << 32) | curand(&RNG);
                else
                    cr &= ((uint64_t)curand(&RNG) << 32) | curand(&RNG);
            }
            return cr;
        }
#endif
    }
}