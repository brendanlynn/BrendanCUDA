#pragma once

#include <cuda_runtime.h>
#include <concepts>
#include <bit>

#include "brendancuda_rand_anyrng.h"
#include "brendancuda_rand_bits.h"

namespace BrendanCUDA {
    namespace details {
        template <std::integral _T>
        __host__ __device__ __forceinline _T RandomizeWTargets_GetEditsOf1s(_T Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t RN1, uint32_t RN2);
    }
    namespace Random {
        template <std::integral _T>
        __host__ __device__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, AnyRNG<uint64_t> RNG);
        template <std::integral _T>
        __host__ __device__ __forceinline _T RandomizeWTargets(_T Value, uint32_t EachFlipProbability, AnyRNG<uint64_t> RNG);
        template <std::integral _T>
        __host__ __device__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, AnyRNG<uint64_t> RNG);
    }
}

template <std::integral _T>
__host__ __device__ __forceinline _T BrendanCUDA::details::RandomizeWTargets_GetEditsOf1s(_T Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t RN1, uint32_t RN2) {
    if (RN1 < FlipProb) {
        uint32_t mrn = RN2 % CountOf1s;

        for (uint32_t m = 1ui32; m; m <<= 1) {
            if (m & Value) {
                if (!mrn) {
                    return m;
                }
                --mrn;
            }
        }
    }
    return 0;
}

template <std::integral _T>
__host__ __device__ __forceinline _T BrendanCUDA::Random::RandomizeWFlips(_T Value, uint32_t FlipProbability, AnyRNG<uint64_t> RNG) {
    Value ^= (_T)Get64Bits(FlipProbability, RNG);
}
template <std::integral _T>
__host__ __device__ __forceinline _T BrendanCUDA::Random::RandomizeWTargets(_T Value, uint32_t FlipProbability, AnyRNG<uint64_t> RNG) {
    constexpr uint32_t shiftMask = (sizeof(_T) << 3) - 1;
    
    uint64_t rn = RNG();
    uint32_t rn1 = rn;
    uint32_t rn2 = rn >> 32;
    if (!Value) {
        if (rn1 < FlipProbability) {
            return ((_T)1) << (shiftMask & rn2);
        }
        return (_T)0;
    }
    else if (Value == ~(_T)0) {
        if (rn1 < FlipProbability) {
            return ~(((_T)1) << (shiftMask & rn2));
        }
        return ~(_T)0;
    }
    else {
        rn = RNG();
        uint32_t rn3 = rn;
        uint32_t rn4 = rn >> 32;

        uint32_t bc = std::popcount(Value);

        return Value ^
            details::RandomizeWTargets_GetEditsOf1s(Value, bc, FlipProbability, rn1, rn2) ^
            details::RandomizeWTargets_GetEditsOf1s(~Value, 64 - bc, FlipProbability, rn3, rn4);
    }
}
template <std::integral _T>
__host__ __device__ __forceinline _T BrendanCUDA::Random::RandomizeWMutations(_T Value, uint32_t MutationProbability, AnyRNG<uint64_t> RNG) {
    if (RNG() < MutationProbability) return (_T)RNG();
}