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
        __host__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, AnyRNG<uint64_t> RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, curandState& State);
#endif
        template <std::integral _T>
        __host__ __forceinline _T RandomizeWTargets(_T Value, uint32_t EachFlipProbability, AnyRNG<uint64_t> RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWTargets(_T Value, uint32_t EachFlipProbability, curandState& State);
#endif
        template <std::integral _T>
        __host__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, AnyRNG<uint64_t> RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, curandState& State);
#endif
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
__host__ __forceinline _T BrendanCUDA::Random::RandomizeWFlips(_T Value, uint32_t FlipProbability, AnyRNG<uint64_t> RNG) {
    if constexpr (sizeof(_T) > 4) Value ^= (_T)Get64Bits(FlipProbability, RNG);
    else Value ^= (_T)Get32Bits(FlipProbability, RNG);
}
#ifdef __CUDACC__
template <std::integral _T>
__device__ __forceinline _T BrendanCUDA::Random::RandomizeWFlips(_T Value, uint32_t FlipProbability, curandState& RNG) {
    if constexpr (sizeof(_T) > 4) Value ^= (_T)Get64Bits(FlipProbability, RNG);
    else Value ^= (_T)Get32Bits(FlipProbability, RNG);
}
#endif
template <std::integral _T>
__host__ __forceinline _T BrendanCUDA::Random::RandomizeWTargets(_T Value, uint32_t FlipProbability, AnyRNG<uint64_t> RNG) {
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
#ifdef __CUDACC__
template <std::integral _T>
__device__ __forceinline _T BrendanCUDA::Random::RandomizeWTargets(_T Value, uint32_t FlipProbability, curandState& RNG) {
    constexpr uint32_t shiftMask = (sizeof(_T) << 3) - 1;

    if (!Value) {
        if (curand(&RNG) < FlipProbability) {
            return ((_T)1) << (shiftMask & curand(&RNG));
        }
        return (_T)0;
    }
    else if (Value == ~(_T)0) {
        if (curand(&RNG) < FlipProbability) {
            return ~(((_T)1) << (shiftMask & curand(&RNG)));
        }
        return ~(_T)0;
    }
    else {
        uint32_t bc = std::popcount(Value);

        return Value ^
            details::RandomizeWTargets_GetEditsOf1s(Value, bc, FlipProbability, curand(&RNG), curand(&RNG)) ^
            details::RandomizeWTargets_GetEditsOf1s(~Value, 64 - bc, FlipProbability, curand(&RNG), curand(&RNG));
    }
}
#endif
template <std::integral _T>
__host__ __forceinline _T BrendanCUDA::Random::RandomizeWMutations(_T Value, uint32_t MutationProbability, AnyRNG<uint64_t> RNG) {
    std::uniform_int_distribution<uint32_t> dis32(0);
    if (dis32(RNG) < MutationProbability) {
        if constexpr (std::is_same<_T, uint32_t>) return dis32(RNG);
        std::uniform_int_distribution<_T> disT(0);
        return disT(RNG);
    }
    return Value;
}
#ifdef __CUDACC__
template <std::integral _T>
__device__ __forceinline _T BrendanCUDA::Random::RandomizeWMutations(_T Value, uint32_t MutationProbability, curandState& RNG) {
    if (curand(&RNG) < MutationProbability) {
        if constexpr (sizeof(_T) > 4) return (_T)(((uint64_t)curand(&RNG) << 32) | curand(&RNG));
        else return (_T)curand(&RNG);
    }
    return Value;
}
#endif