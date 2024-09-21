#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>

namespace brendancuda {
    namespace details {
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ static __forceinline _T GetIntBin(_TRNG& RNG) {
            if constexpr (sizeof(_T) >= 4) {
                std::uniform_int_distribution<std::make_unsigned_t<_T>> dis(0);
                return (_T)dis(RNG);
            }
            else {
                std::uniform_int_distribution<uint32_t> dis32(0, (1u << (sizeof(_T) << 3)) - 1);
                return (_T)dis32(RNG);
            }
        }
#ifdef __CUDACC__
        template <std::integral _T, KernelCurandState _TRNG>
        __device__ static __forceinline _T GetIntBin(_TRNG& RNG) {
            if constexpr (sizeof(_T) == 8) {
                return ((_T)curand(RNG) << 32) | (_T)curand(RNG);
            }
            else {
                return (_T)curand(RNG);
            }
        }
#endif
    }
}