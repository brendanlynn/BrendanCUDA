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
        void InitArray_CallKernel(Span<float> Array, uint64_t Seed);
        void InitArray_CallKernel(Span<double> Array, uint64_t Seed);
        void InitArray_CallKernel(Span<float> Array, float LowerBound, float UpperBound, uint64_t Seed);
        void InitArray_CallKernel(Span<double> Array, double LowerBound, double UpperBound, uint64_t Seed);
        void InitArray_CallKernel(Span<uint32_t> Array, uint64_t Seed);
        void InitArray_CallKernel(Span<uint32_t> Array, uint32_t ProbabilityOf1, uint64_t Seed);
        void ClearArray_CallKernel(Span<float> Array);
        void ClearArray_CallKernel(Span<double> Array);
        void ClearArray_CallKernel(Span<uint64_t> Array);
    }
    namespace Random {
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, _TRNG& RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, curandState& RNG);
#endif
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWTargets(_T Value, uint32_t EachFlipProbability, _TRNG& RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWTargets(_T Value, uint32_t EachFlipProbability, curandState& RNG);
#endif
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, _TRNG& RNG);
#ifdef __CUDACC__
        template <std::integral _T>
        __device__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, curandState& RNG);
#endif
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline float Randomize(float Value, float Scalar, _TRNG& RNG);
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline double Randomize(double Value, double Scalar, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ __forceinline float Randomize(float Value, float Scalar, curandState& RNG);
        __device__ __forceinline double Randomize(double Value, double Scalar, curandState& RNG);
#endif
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline float Randomize(float Value, float Scalar, float LowerBound, float UpperBound, _TRNG& RNG);
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline double Randomize(double Value, double Scalar, double LowerBound, double UpperBound, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ __forceinline float Randomize(float Value, float Scalar, float LowerBound, float UpperBound, curandState& RNG);
        __device__ __forceinline double Randomize(double Value, double Scalar, double LowerBound, double UpperBound, curandState& RNG);
#endif

        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArray(Span<float> Array, float Scalar, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void RandomizeArray(Span<float> Array, float Scalar, curandState& RNG);
#endif
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArray(Span<float> Array, float Scalar, float LowerBound, float UpperBound, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void RandomizeArray(Span<float> Array, float Scalar, float LowerBound, float UpperBound, curandState& RNG);
#endif

        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArray(Span<double> Array, double Scalar, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void RandomizeArray(Span<double> Array, double Scalar, curandState& RNG);
#endif
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArray(Span<double> Array, double Scalar, double LowerBound, double UpperBound, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void RandomizeArray(Span<double> Array, double Scalar, double LowerBound, double UpperBound, curandState& RNG);
#endif

        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWFlips(Span<uint64_t> Array, uint32_t FlipProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWFlips(Span<uint32_t> Array, uint32_t FlipProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWFlips(Span<uint16_t> Array, uint32_t FlipProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWFlips(Span<uint8_t> Array, uint32_t FlipProb, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void RandomizeArrayWFlips(Span<uint64_t> Array, uint32_t FlipProb, curandState& RNG);
        __device__ void RandomizeArrayWFlips(Span<uint32_t> Array, uint32_t FlipProb, curandState& RNG);
        __device__ void RandomizeArrayWFlips(Span<uint16_t> Array, uint32_t FlipProb, curandState& RNG);
        __device__ void RandomizeArrayWFlips(Span<uint8_t> Array, uint32_t FlipProb, curandState& RNG);
#endif
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWTargets(Span<uint64_t> Array, uint32_t EachFlipProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWTargets(Span<uint32_t> Array, uint32_t EachFlipProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWTargets(Span<uint16_t> Array, uint32_t EachFlipProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWTargets(Span<uint8_t> Array, uint32_t EachFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void RandomizeArrayWTargets(Span<uint64_t> Array, uint32_t EachFlipProb, curandState& RNG);
        __device__ void RandomizeArrayWTargets(Span<uint32_t> Array, uint32_t EachFlipProb, curandState& RNG);
        __device__ void RandomizeArrayWTargets(Span<uint16_t> Array, uint32_t EachFlipProb, curandState& RNG);
        __device__ void RandomizeArrayWTargets(Span<uint8_t> Array, uint32_t EachFlipProb, curandState& RNG);
#endif
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWMutations(Span<uint64_t> Array, uint32_t MutationProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWMutations(Span<uint32_t> Array, uint32_t MutationProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWMutations(Span<uint16_t> Array, uint32_t MutationProb, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void RandomizeArrayWMutations(Span<uint8_t> Array, uint32_t MutationProb, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void RandomizeArrayWMutations(Span<uint64_t> Array, uint32_t MutationProb, curandState& RNG);
        __device__ void RandomizeArrayWMutations(Span<uint32_t> Array, uint32_t MutationProb, curandState& RNG);
        __device__ void RandomizeArrayWMutations(Span<uint16_t> Array, uint32_t MutationProb, curandState& RNG);
        __device__ void RandomizeArrayWMutations(Span<uint8_t> Array, uint32_t MutationProb, curandState& RNG);
#endif

        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<float> Array, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void InitRandomArray(Span<float> Array, curandState& RNG);
#endif
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<float> Array, float LowerBound, float UpperBound, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void InitRandomArray(Span<float> Array, float LowerBound, float UpperBound, curandState& RNG);
#endif

        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<double> Array, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void InitRandomArray(Span<double> Array, curandState& RNG);
#endif
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<double> Array, double LowerBound, double UpperBound, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void InitRandomArray(Span<double> Array, double LowerBound, double UpperBound, curandState& RNG);
#endif

        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint64_t> Array, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint32_t> Array, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint16_t> Array, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint8_t> Array, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void InitRandomArray(Span<uint64_t> Array, curandState& RNG);
        __device__ void InitRandomArray(Span<uint32_t> Array, curandState& RNG);
        __device__ void InitRandomArray(Span<uint16_t> Array, curandState& RNG);
        __device__ void InitRandomArray(Span<uint8_t> Array, curandState& RNG);
#endif

        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint64_t> Array, uint32_t ProbabilityOf1, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint32_t> Array, uint32_t ProbabilityOf1, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint16_t> Array, uint32_t ProbabilityOf1, _TRNG& RNG);
        template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
        __host__ void InitRandomArray(Span<uint8_t> Array, uint32_t ProbabilityOf1, _TRNG& RNG);
#ifdef __CUDACC__
        __device__ void InitRandomArray(Span<uint64_t> Array, uint32_t ProbabilityOf1, curandState& RNG);
        __device__ void InitRandomArray(Span<uint32_t> Array, uint32_t ProbabilityOf1, curandState& RNG);
        __device__ void InitRandomArray(Span<uint16_t> Array, uint32_t ProbabilityOf1, curandState& RNG);
        __device__ void InitRandomArray(Span<uint8_t> Array, uint32_t ProbabilityOf1, curandState& RNG);
#endif

        template <bool _MemoryOnHost>
        __host__ void ClearArray(Span<float> Array);
        __device__ void ClearArray(Span<float> Array);
        template <bool _MemoryOnHost>
        __host__ void ClearArray(Span<double> Array);
        __device__ void ClearArray(Span<double> Array);

        template <bool _MemoryOnHost>
        __host__ void ClearArray(Span<uint64_t> Array);
        template <bool _MemoryOnHost>
        __host__ void ClearArray(Span<uint32_t> Array);
        template <bool _MemoryOnHost>
        __host__ void ClearArray(Span<uint16_t> Array);
        template <bool _MemoryOnHost>
        __host__ void ClearArray(Span<uint8_t> Array);
#ifdef __CUDACC__
        __device__ void ClearArray(Span<uint64_t> Array);
        __device__ void ClearArray(Span<uint32_t> Array);
        __device__ void ClearArray(Span<uint16_t> Array);
        __device__ void ClearArray(Span<uint8_t> Array);
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
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline float BrendanCUDA::Random::Randomize(float Value, float Scalar, _TRNG& RNG) {
    std::uniform_real_distribution<float> dis(-Scalar, Scalar);
    return Value + dis(RNG);
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline double BrendanCUDA::Random::Randomize(double Value, double Scalar, _TRNG& RNG) {
    std::uniform_real_distribution<double> dis(-Scalar, Scalar);
    return Value + dis(RNG);
}
#ifdef __CUDACC__
__device__ __forceinline float BrendanCUDA::Random::Randomize(float Value, float Scalar, curandState& RNG) {
    return Value + Scalar * 2.f * (curand_uniform(&RNG) - 0.5f);
}
__device__ __forceinline double BrendanCUDA::Random::Randomize(double Value, double Scalar, curandState& RNG) {
    return Value + Scalar * 2. * (curand_uniform_double(&RNG) - 0.5);
}
#endif
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline float BrendanCUDA::Random::Randomize(float Value, float Scalar, float LowerBound, float UpperBound, _TRNG& RNG) {
    std::uniform_real_distribution<float> dis(-Scalar, Scalar);
    return std::clamp(Value + dis(RNG), LowerBound, UpperBound);
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline double BrendanCUDA::Random::Randomize(double Value, double Scalar, double LowerBound, double UpperBound, _TRNG& RNG) {
    std::uniform_real_distribution<double> dis(-Scalar, Scalar);
    return std::clamp(Value + dis(RNG), LowerBound, UpperBound);
}
#ifdef __CUDACC__
__device__ __forceinline float BrendanCUDA::Random::Randomize(float Value, float Scalar, float LowerBound, float UpperBound, curandState& RNG) {
    return std::clamp(Value + Scalar * 2.f * (curand_uniform(&RNG) - 0.5f), LowerBound, UpperBound);
}
__device__ __forceinline double BrendanCUDA::Random::Randomize(double Value, double Scalar, double LowerBound, double UpperBound, curandState& RNG) {
    return std::clamp(Value + Scalar * 2. * (curand_uniform_double(&RNG) - 0.5), LowerBound, UpperBound);
}
#endif

template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::RandomizeArray(Span<float> Array, float Scalar, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        Scalar *= 2.f;
        std::uniform_real_distribution<float> dis(-Scalar, Scalar);
        float* l = Array.ptr;
        float* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l += dis(RNG);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::RandomizeArray_CallKernel(Array, Scalar, dis64(RNG));
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::RandomizeArray(Span<float> Array, float Scalar, curandState& RNG) {
    Scalar *= 2.f;
    float* l = Array.ptr;
    float* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l += Scalar * (curand_uniform(&RNG) - 0.5f);
}
#endif
template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::RandomizeArray(Span<float> Array, float Scalar, float LowerBound, float UpperBound, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        Scalar *= 2.f;
        std::uniform_real_distribution<float> dis(-Scalar, Scalar);
        float* l = Array.ptr;
        float* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l = std::clamp(*l + dis(RNG), LowerBound, UpperBound);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::RandomizeArray_CallKernel(Array, Scalar, LowerBound, UpperBound, dis64(RNG));
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::RandomizeArray(Span<float> Array, float Scalar, float LowerBound, float UpperBound, curandState& RNG) {
    Scalar *= 2.f;
    float* l = Array.ptr;
    float* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l = std::clamp(*l + Scalar * (curand_uniform(&RNG) - 0.5f), LowerBound, UpperBound);
}
#endif

template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::RandomizeArray(Span<double> Array, double Scalar, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        Scalar *= 2.f;
        std::uniform_real_distribution<double> dis(-Scalar, Scalar);
        double* l = Array.ptr;
        double* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l += dis(RNG);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::RandomizeArray_CallKernel(Array, Scalar, dis64(RNG));
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::RandomizeArray(Span<double> Array, double Scalar, curandState& RNG) {
    Scalar *= 2.f;
    double* l = Array.ptr;
    double* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l += Scalar * (curand_uniform_double(&RNG) - 0.5);
}
#endif
template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::RandomizeArray(Span<double> Array, double Scalar, double LowerBound, double UpperBound, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        Scalar *= 2.f;
        std::uniform_real_distribution<double> dis(-Scalar, Scalar);
        double* l = Array.ptr;
        double* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l = std::clamp(*l + dis(RNG), LowerBound, UpperBound);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::RandomizeArray_CallKernel(Array, Scalar, LowerBound, UpperBound, dis64(RNG));
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::RandomizeArray(Span<double> Array, double Scalar, double LowerBound, double UpperBound, curandState& RNG) {
    Scalar *= 2.f;
    double* l = Array.ptr;
    double* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l = std::clamp(*l + Scalar * (curand_uniform_double(&RNG) - 0.5), LowerBound, UpperBound);
}
#endif

template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::RandomizeArrayWFlips(Span<uint8_t> Array, uint32_t FlipProb, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        uint64_t* l64 = (uint64_t*)Array.ptr;
        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
        for (; l64 < u64; ++l64) *l64 = RandomizeWFlips(*l64, FlipProb, RNG);
        
        uint64_t endV;
        size_t r = Array.size & 7;
        memcpy(&endV, u64, r);
        if (r > 4) endV = RandomizeWFlips(endV, FlipProb, RNG);
        else if (r > 2) endV = (uint64_t)RandomizeWFlips((uint32_t)endV, FlipProb, RNG);
        else if (r > 1) endV = (uint64_t)RandomizeWFlips((uint16_t)endV, FlipProb, RNG);
        else endV = (uint64_t)RandomizeWFlips((uint8_t)endV, FlipProb, RNG);
        memcpy(u64, &endV, r & 7);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::RandomizeArray_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), FlipProb, dis64(RNG));

        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

        uint64_t endV;
        size_t r = Array.size & 7;
        cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
        if (r > 4) endV = RandomizeWFlips(endV, FlipProb, RNG);
        else if (r > 2) endV = (uint64_t)RandomizeWFlips((uint32_t)endV, FlipProb, RNG);
        else if (r > 1) endV = (uint64_t)RandomizeWFlips((uint16_t)endV, FlipProb, RNG);
        else endV = (uint64_t)RandomizeWFlips((uint8_t)endV, FlipProb, RNG);
        cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::RandomizeArrayWFlips(Span<uint8_t> Array, uint32_t FlipProb, curandState& RNG) {
    uint64_t* l64 = (uint64_t*)Array.ptr;
    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
    for (; l64 < u64; ++l64) *l64 = RandomizeWFlips(*l64, FlipProb, RNG);

    uint64_t endV;
    size_t r = Array.size & 7;
    memcpy(&endV, u64, r);
    if (r > 4) endV = RandomizeWFlips(endV, FlipProb, RNG);
    else if (r > 2) endV = (uint64_t)RandomizeWFlips((uint32_t)endV, FlipProb, RNG);
    else if (r > 1) endV = (uint64_t)RandomizeWFlips((uint16_t)endV, FlipProb, RNG);
    else endV = (uint64_t)RandomizeWFlips((uint8_t)endV, FlipProb, RNG);
    memcpy(u64, &endV, r);
}
#endif
template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::RandomizeArrayWTargets(Span<uint8_t> Array, uint32_t EachFlipProb, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        uint64_t* l64 = (uint64_t*)Array.ptr;
        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
        for (; l64 < u64; ++l64) *l64 = RandomizeWTargets(*l64, EachFlipProb, RNG);

        uint64_t endV;
        size_t r = Array.size & 7;
        memcpy(&endV, u64, r);
        if (r > 4) endV = RandomizeWTargets(endV, EachFlipProb, RNG);
        else if (r > 2) endV = (uint64_t)RandomizeWTargets((uint32_t)endV, EachFlipProb, RNG);
        else if (r > 1) endV = (uint64_t)RandomizeWTargets((uint16_t)endV, EachFlipProb, RNG);
        else endV = (uint64_t)RandomizeWTargets((uint8_t)endV, EachFlipProb, RNG);
        memcpy(u64, &endV, r);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::RandomizeArray_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), EachFlipProb, dis64(RNG));

        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

        uint64_t endV;
        size_t r = Array.size & 7;
        cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
        if (r > 4) endV = RandomizeWTargets(endV, EachFlipProb, RNG);
        else if (r > 2) endV = (uint64_t)RandomizeWTargets((uint32_t)endV, EachFlipProb, RNG);
        else if (r > 1) endV = (uint64_t)RandomizeWTargets((uint16_t)endV, EachFlipProb, RNG);
        else endV = (uint64_t)RandomizeWTargets((uint8_t)endV, EachFlipProb, RNG);
        cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::RandomizeArrayWTargets(Span<uint8_t> Array, uint32_t EachFlipProb, curandState& RNG) {
    uint64_t* l64 = (uint64_t*)Array.ptr;
    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
    for (; l64 < u64; ++l64) *l64 = RandomizeWTargets(*l64, EachFlipProb, RNG);

    uint64_t endV;
    size_t r = Array.size & 7;
    memcpy(&endV, u64, r);
    if (r > 4) endV = RandomizeWTargets(endV, EachFlipProb, RNG);
    else if (r > 2) endV = (uint64_t)RandomizeWTargets((uint32_t)endV, EachFlipProb, RNG);
    else if (r > 1) endV = (uint64_t)RandomizeWTargets((uint16_t)endV, EachFlipProb, RNG);
    else endV = (uint64_t)RandomizeWTargets((uint8_t)endV, EachFlipProb, RNG);
    memcpy(u64, &endV, r);
}
#endif
template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::RandomizeArrayWMutations(Span<uint8_t> Array, uint32_t MutationProb, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        uint64_t* l64 = (uint64_t*)Array.ptr;
        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
        for (; l64 < u64; ++l64) *l64 = RandomizeWMutations(*l64, MutationProb, RNG);

        uint64_t endV;
        size_t r = Array.size & 7;
        memcpy(&endV, u64, r);
        if (r > 4) endV = RandomizeWMutations(endV, MutationProb, RNG);
        else if (r > 2) endV = (uint64_t)RandomizeWMutations((uint32_t)endV, MutationProb, RNG);
        else if (r > 1) endV = (uint64_t)RandomizeWMutations((uint16_t)endV, MutationProb, RNG);
        else endV = (uint64_t)RandomizeWMutations((uint8_t)endV, MutationProb, RNG);
        memcpy(u64, &endV, r);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::RandomizeArray_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), MutationProb, dis64(RNG));

        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

        uint64_t endV;
        size_t r = Array.size & 7;
        cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
        if (r > 4) endV = RandomizeWMutations(endV, MutationProb, RNG);
        else if (r > 2) endV = (uint64_t)RandomizeWMutations((uint32_t)endV, MutationProb, RNG);
        else if (r > 1) endV = (uint64_t)RandomizeWMutations((uint16_t)endV, MutationProb, RNG);
        else endV = (uint64_t)RandomizeWMutations((uint8_t)endV, MutationProb, RNG);
        cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::RandomizeArrayWMutations(Span<uint8_t> Array, uint32_t MutationProb, curandState& RNG) {
    uint64_t* l64 = (uint64_t*)Array.ptr;
    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
    for (; l64 < u64; ++l64) *l64 = RandomizeWMutations(*l64, MutationProb, RNG);

    uint64_t endV;
    size_t r = Array.size & 7;
    memcpy(&endV, u64, r);
    if (r > 4) endV = RandomizeWMutations(endV, MutationProb, RNG);
    else if (r > 2) endV = (uint64_t)RandomizeWMutations((uint32_t)endV, MutationProb, RNG);
    else if (r > 1) endV = (uint64_t)RandomizeWMutations((uint16_t)endV, MutationProb, RNG);
    else endV = (uint64_t)RandomizeWMutations((uint8_t)endV, MutationProb, RNG);
    memcpy(u64, &endV, r);
}
#endif

template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::InitRandomArray(Span<float> Array, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        std::uniform_real_distribution<float> dis(-1.f, 1.f);
        float* l = Array.ptr;
        float* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l = dis(RNG);
        break;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::InitArray_CallKernel(Array, dis64(RNG));
        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::InitRandomArray(Span<float> Array, curandState& RNG) {
    float* l = Array.ptr;
    float* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l = curand_uniform(&RNG);
}
#endif
template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::InitRandomArray(Span<float> Array, float LowerBound, float UpperBound, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        std::uniform_real_distribution<float> dis(LowerBound, UpperBound);
        float* l = Array.ptr;
        float* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l = dis(RNG);
        break;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::InitArray_CallKernel(Array, LowerBound, UpperBound, dis64(RNG));
        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::InitRandomArray(Span<float> Array, float LowerBound, float UpperBound, curandState& RNG) {
    float range = UpperBound - LowerBound;
    float* l = Array.ptr;
    float* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l = curand_uniform(&RNG) * range + LowerBound;
}
#endif

template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::InitRandomArray(Span<double> Array, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        std::uniform_real_distribution<double> dis(-1.f, 1.f);
        double* l = Array.ptr;
        double* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l = dis(RNG);
        break;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::InitArray_CallKernel(Array, dis64(RNG));
        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::InitRandomArray(Span<double> Array, curandState& RNG) {
    double* l = Array.ptr;
    double* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l = curand_uniform_double(&RNG);
}
#endif
template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::InitRandomArray(Span<double> Array, double LowerBound, double UpperBound, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        std::uniform_real_distribution<double> dis(LowerBound, UpperBound);
        double* l = Array.ptr;
        double* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l = dis(RNG);
        break;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::InitArray_CallKernel(Array, LowerBound, UpperBound, dis64(RNG));
        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::InitRandomArray(Span<double> Array, double LowerBound, double UpperBound, curandState& RNG) {
    double range = UpperBound - LowerBound;
    double* l = Array.ptr;
    double* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l = curand_uniform_double(&RNG) * range + LowerBound;
}
#endif

template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::InitRandomArray(Span<uint8_t> Array, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        std::uniform_int_distribution<uint64_t> dis64(0);
        uint64_t* l64 = (uint64_t*)Array.ptr;
        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
        for (; l64 < u64; ++l64) *l64 = dis64(RNG);

        uint64_t endV;
        size_t r = Array.size & 7;
        memcpy(&endV, u64, r);
        endV = dis64(RNG);
        memcpy(u64, &endV, r);

        break;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::InitArray_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), MutationProb, dis64(RNG));

        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

        uint64_t endV;
        size_t r = Array.size & 7;
        cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
        endV = dis64(RNG);
        cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);

        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::InitRandomArray(Span<uint8_t> Array, curandState& RNG) {
    uint32_t* l32 = (uint32_t*)Array.ptr;
    uint32_t* u32 = ((uint32_t*)Array.ptr) + (Array.size >> 2);
    for (; l32 < u32; ++l32) *l32 = curand(&RNG);

    uint32_t endV;
    size_t r = Array.size & 3;
    memcpy(&endV, u32, r);
    endV = curand(&RNG);
    memcpy(u32, &endV, r);
}
#endif

template <bool _MemoryOnHost, std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::Random::InitRandomArray(Span<uint8_t> Array, uint32_t ProbabilityOf1, _TRNG& RNG) {
    if constexpr (_MemoryOnHost) {
        uint64_t* l64 = (uint64_t*)Array.ptr;
        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
        for (; l64 < u64; ++l64) *l64 = Get64Bits(ProbabilityOf1, RNG);

        uint64_t endV;
        size_t r = Array.size & 7;
        memcpy(&endV, u64, r);
        endV = Get64Bits(ProbabilityOf1, RNG);
        memcpy(u64, &endV, r);

        break;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::InitArray_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), MutationProb, dis64(RNG));

        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

        uint64_t endV;
        size_t r = Array.size & 7;
        cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
        endV = Get64Bits(ProbabilityOf1, RNG);
        cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);

        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::InitRandomArray(Span<uint8_t> Array, uint32_t ProbabilityOf1, curandState& RNG) {
    uint32_t* l32 = (uint32_t*)Array.ptr;
    uint32_t* u32 = ((uint32_t*)Array.ptr) + (Array.size >> 2);
    for (; l32 < u32; ++l64) *l32 = Get32Bits(ProbabilityOf1, RNG);

    uint32_t endV;
    size_t r = Array.size & 3;
    memcpy(&endV, u32, r);
    endV = Get32Bits(ProbabilityOf1, RNG);
    memcpy(u32, &endV, r);
}
#endif

template <bool _MemoryOnHost>
__host__ void BrendanCUDA::Random::ClearArray(Span<float> Array) {
    if constexpr (_MemoryOnHost) {
        float* l = Array.ptr;
        float* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l += 0.f;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::ClearArray_CallKernel(Array);
    }
}
__device__ void BrendanCUDA::Random::ClearArray(Span<float> Array) {
    float* l = Array.ptr;
    float* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l += 0.f;
}
template <bool _MemoryOnHost>
__host__ void BrendanCUDA::Random::ClearArray(Span<double> Array) {
    if constexpr (_MemoryOnHost) {
        double* l = Array.ptr;
        double* u = Array.ptr + Array.size;
        for (; l < u; ++l) *l += 0.;
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::ClearArray_CallKernel(Array);
    }
}
__device__ void BrendanCUDA::Random::ClearArray(Span<double> Array) {
    double* l = Array.ptr;
    double* u = Array.ptr + Array.size;
    for (; l < u; ++l) *l += 0.;
}
template <bool _MemoryOnHost>
__host__ void BrendanCUDA::Random::ClearArray(Span<uint8_t> Array) {
    if constexpr (_MemoryOnHost) {
        uint64_t* l64 = (uint64_t*)Array.ptr;
        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
        for (; l64 < u64; ++l64) *l64 = 0;

        uint64_t endV = 0;
        size_t r = Array.size & 7;
        memcpy(u64, &endV, r);
    }
    else {
        std::uniform_int_distribution<uint64_t> dis64(0);
        details::ClearArray_CallKernel(Span<uint64_t>((uint64_t*)Array.ptr, Array.size >> 3));

        uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

        uint64_t endV = 0;
        size_t r = Array.size & 7;
        cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::Random::ClearArray(Span<uint8_t> Array) {
    uint64_t* l64 = (uint64_t*)Array.ptr;
    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
    for (; l64 < u64; ++l64) *l64 = 0;

    uint64_t endV = 0;
    size_t r = Array.size & 7;
    memcpy(u64, &endV, r);
}}
#endif