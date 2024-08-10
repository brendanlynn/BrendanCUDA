#pragma once

#include <cuda_runtime.h>
#include <concepts>
#include <bit>
#include <curand_kernel.h>

#include "brendancuda_rand_anyrng.h"
#include "brendancuda_rand_bits.h"
#include "brendancuda_arrays.h"

namespace BrendanCUDA {
    namespace details {
        template <std::integral _T>
        __host__ __device__ __forceinline _T RandomizeWTargets_GetEditsOf1s(_T Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t RN1, uint32_t RN2);
        
        void RandomizeArray_CallKernel(Span<float> Array, float Scalar, uint64_t Seed);
        void RandomizeArray_CallKernel(Span<double> Array, double Scalar, uint64_t Seed);
        void RandomizeArray_CallKernel(Span<float> Array, float Scalar, float LowerBound, float UpperBound, uint64_t Seed);
        void RandomizeArray_CallKernel(Span<double> Array, double Scalar, double LowerBound, double UpperBound, uint64_t Seed);
        void RandomizeArrayWFlips_CallKernel(Span<uint32_t> Array, uint32_t FlipProbability, uint64_t Seed);
        void RandomizeArrayWTargets_CallKernel(Span<uint32_t> Array, uint32_t EachFlipProbability, uint64_t Seed);
        void RandomizeArrayWMutations_CallKernel(Span<uint32_t> Array, uint32_t MutationProbability, uint64_t Seed);
        void RandomizeArray_CallKernel(Span<uint32_t> Array, uint32_t FlipProbability, uint32_t TargetFlipProbability, uint32_t MutationProbability, uint64_t Seed);
        void InitArray_CallKernel(Span<float> Array, uint64_t Seed);
        void InitArray_CallKernel(Span<double> Array, uint64_t Seed);
        void InitArray_CallKernel(Span<float> Array, float LowerBound, float UpperBound, uint64_t Seed);
        void InitArray_CallKernel(Span<double> Array, double LowerBound, double UpperBound, uint64_t Seed);
        void InitArray_CallKernel(Span<uint32_t> Array, uint64_t Seed);
        void InitArray_CallKernel(Span<uint32_t> Array, uint32_t ProbabilityOf1, uint64_t Seed);
    }
    namespace Random {
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, _TRNG& RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, curandState& State);
#endif
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWTargets(_T Value, uint32_t EachFlipProbability, _TRNG& RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWTargets(_T Value, uint32_t EachFlipProbability, curandState& State);
#endif
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, _TRNG& RNG);
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

template <std::integral _T, std::uniform_random_bit_generator _TRNG>
__host__ __forceinline _T BrendanCUDA::Random::RandomizeWFlips(_T Value, uint32_t FlipProbability, _TRNG& RNG) {
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
template <std::integral _T, std::uniform_random_bit_generator _TRNG>
__host__ __forceinline _T BrendanCUDA::Random::RandomizeWTargets(_T Value, uint32_t FlipProbability, _TRNG& RNG) {
    constexpr uint32_t shiftMask = (sizeof(_T) << 3) - 1;
    
    std::uniform_int_distribution<uint32_t> dis32(0);

    if (!Value) {
        if (dis32(RNG) < FlipProbability) {
            return ((_T)1) << (shiftMask & dis32(RNG));
        }
        return (_T)0;
    }
    else if (Value == ~(_T)0) {
        if (dis32(RNG) < FlipProbability) {
            return ~(((_T)1) << (shiftMask & dis32(RNG)));
        }
        return ~(_T)0;
    }
    else {
        uint32_t bc = std::popcount(Value);

        return Value ^
            details::RandomizeWTargets_GetEditsOf1s(Value, bc, FlipProbability, dis32(RNG), dis32(RNG)) ^
            details::RandomizeWTargets_GetEditsOf1s(~Value, 64 - bc, FlipProbability, dis32(RNG), dis32(RNG));
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
template <std::integral _T, std::uniform_random_bit_generator _TRNG>
__host__ __forceinline _T BrendanCUDA::Random::RandomizeWMutations(_T Value, uint32_t MutationProbability, _TRNG& RNG) {
    std::uniform_int_distribution<uint32_t> dis32(0);
    if (dis32(RNG) < MutationProbability) {
        if constexpr (std::same_as<_T, uint32_t>) return dis32(RNG);
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