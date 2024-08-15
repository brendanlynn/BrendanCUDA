#pragma once

#include <random>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace details {
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        static __forceinline _T GetIntBin(_TRNG& RNG) {
            if constexpr (sizeof(_T) >= 4) {
                std::uniform_int_distribution<std::make_unsigned_t<_T>> dis(0);
                return (_T)dis(RNG);
            }
            else {
                std::uniform_int_distribution<uint32_t> dis32(0, (1u << (sizeof(_T) << 3)) - 1);
                return (_T)dis32(RNG);
            }
        }
    }
}