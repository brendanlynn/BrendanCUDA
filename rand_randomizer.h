#pragma once

#include "arrays.h"
#include "curandkernelgens.h"
#include "rand_anyrng.h"
#include "rand_bits.h"
#include <bit>
#include <concepts>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace bcuda {
    namespace details {
        template <std::integral _T>
        __host__ __device__ static __forceinline _T RandomizeWTargets_GetEditsOf1s(_T Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t RN1, uint32_t RN2) {
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

        void RandomizeArray_CallKernel(Span<float> Array, float Scalar, uint64_t Seed);
        void RandomizeArray_CallKernel(Span<double> Array, double Scalar, uint64_t Seed);
        void RandomizeArray_CallKernel(Span<float> Array, float Scalar, float LowerBound, float UpperBound, uint64_t Seed);
        void RandomizeArray_CallKernel(Span<double> Array, double Scalar, double LowerBound, double UpperBound, uint64_t Seed);
        void RandomizeArrayWFlips_CallKernel(Span<uint32_t> Array, uint32_t FlipProbability, uint64_t Seed);
        void RandomizeArrayWTargets_CallKernel(Span<uint32_t> Array, uint32_t EachFlipProbability, uint64_t Seed);
        void RandomizeArrayWMutations_CallKernel(Span<uint32_t> Array, uint32_t MutationProbability, uint64_t Seed);
        void RandomizeArrayWMutations_CallKernel(Span<uint32_t> Array, uint32_t MutationProbability, uint32_t ProbabilityOf1, uint64_t Seed);
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
    namespace rand {
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, _TRNG& RNG) {
            if constexpr (sizeof(_T) > 4) return Value ^ (_T)Get64Bits(FlipProbability, RNG);
            else return Value ^ (_T)Get32Bits(FlipProbability, RNG);
    }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline _T RandomizeWFlips(_T Value, uint32_t FlipProbability, _TRNG& RNG) {
            if constexpr (sizeof(_T) > 4) return Value ^ (_T)Get64Bits(FlipProbability, RNG);
            else return Value ^ (_T)Get32Bits(FlipProbability, RNG);
}
#endif
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWTargets(_T Value, uint32_t FlipProbability, _TRNG& RNG) {
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
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline _T RandomizeWTargets(_T Value, uint32_t FlipProbability, _TRNG& RNG) {
            constexpr uint32_t shiftMask = (sizeof(_T) << 3) - 1;

            if (!Value) {
                if (curand(&RNG) < FlipProbability) {
                    return ((_T)1) << (shiftMask & curand(&RNG));
                }
                return (_T)0;
            }
            else if (!~Value) {
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
        __host__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, _TRNG& RNG) {
            std::uniform_int_distribution<uint32_t> dis32(0);
            if (dis32(RNG) < MutationProbability) {
                if constexpr (sizeof(_T) == 4) return (_T)dis32(RNG);
                else if constexpr (sizeof(_T) > 4) {
                    std::uniform_int_distribution<_T> disT(std::numeric_limits<_T>::min(), std::numeric_limits<_T>::max());
                    return disT(RNG);
                }
                else {
                    std::uniform_int_distribution<uint16_t> dis(0, (1 << (sizeof(_T) << 3)) - 1);
                    return (_T)dis(RNG);
                }
            }
            return Value;
        }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, _TRNG& RNG) {
            if (curand(&RNG) < MutationProbability) {
                if constexpr (sizeof(_T) > 4) return (_T)(((uint64_t)curand(&RNG) << 32) | curand(&RNG));
                else return (_T)curand(&RNG);
            }
            return Value;
        }
#endif
        template <std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, uint32_t ProbabilityOf1, _TRNG& RNG) {
            std::uniform_int_distribution<uint32_t> dis32(0);
            if (dis32(RNG) < MutationProbability) {
                if constexpr (sizeof(_T) > 4) return (_T)Get64Bits(ProbabilityOf1, RNG);
                else return (_T)Get32Bits(ProbabilityOf1, RNG);
        }
            return Value;
        }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline _T RandomizeWMutations(_T Value, uint32_t MutationProbability, uint32_t ProbabilityOf1, _TRNG& RNG) {
            if (curand(&RNG) < MutationProbability) {
                if constexpr (sizeof(_T) > 4) return (_T)Get64Bits(ProbabilityOf1, RNG);
                else return (_T)Get32Bits(ProbabilityOf1, RNG);
            }
            return Value;
        }
#endif
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline float Randomize(float Value, float Scalar, _TRNG& RNG) {
            std::uniform_real_distribution<float> dis(-Scalar, Scalar);
            return Value + dis(RNG);
        }
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline double Randomize(double Value, double Scalar, _TRNG& RNG) {
            std::uniform_real_distribution<double> dis(-Scalar, Scalar);
            return Value + dis(RNG);
        }
#ifdef __CUDACC__
        template <bcuda::KernelCurandState _TRNG>
        __device__ __forceinline float Randomize(float Value, float Scalar, _TRNG& RNG) {
            return Value + Scalar * 2.f * (curand_uniform(&RNG) - 0.5f);
        }
        template <bcuda::KernelCurandState _TRNG>
        __device__ __forceinline double Randomize(double Value, double Scalar, _TRNG& RNG) {
            return Value + Scalar * 2. * (curand_uniform_double(&RNG) - 0.5);
        }
#endif
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline float Randomize(float Value, float Scalar, float LowerBound, float UpperBound, _TRNG& RNG) {
            std::uniform_real_distribution<float> dis(-Scalar, Scalar);
            return std::clamp(Value + dis(RNG), LowerBound, UpperBound);
        }
        template <std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline double Randomize(double Value, double Scalar, double LowerBound, double UpperBound, _TRNG& RNG) {
            std::uniform_real_distribution<double> dis(-Scalar, Scalar);
            return std::clamp(Value + dis(RNG), LowerBound, UpperBound);
        }
#ifdef __CUDACC__
        template <bcuda::KernelCurandState _TRNG>
        __device__ __forceinline float Randomize(float Value, float Scalar, float LowerBound, float UpperBound, _TRNG& RNG) {
            return std::clamp(Value + Scalar * 2.f * (curand_uniform(&RNG) - 0.5f), LowerBound, UpperBound);
        }
        template <bcuda::KernelCurandState _TRNG>
        __device__ __forceinline double Randomize(double Value, double Scalar, double LowerBound, double UpperBound, _TRNG& RNG) {
            return std::clamp(Value + Scalar * 2. * (curand_uniform_double(&RNG) - 0.5), LowerBound, UpperBound);
        }
#endif

        template <bool _MemoryOnHost, std::floating_point _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void RandomizeArray(Span<_T> Array, _T Scalar, _TRNG& RNG) {
            if constexpr (_MemoryOnHost) {
                Scalar *= 2.f;
                std::uniform_real_distribution<_T> dis(-Scalar, Scalar);
                _T* l = Array.ptr;
                _T* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l += dis(RNG);
            }
            else {
                std::uniform_int_distribution<uint64_t> dis64(0);
                details::RandomizeArray_CallKernel(Array, Scalar, dis64(RNG));
            }
        }
#ifdef __CUDACC__
        template <std::floating_point _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void RandomizeArray(Span<_T> Array, _T Scalar, _TRNG& RNG) {
            if constexpr (std::same_as<_T, float>) {
                Scalar *= 2.f;
                float* l = Array.ptr;
                float* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l += Scalar * (curand_uniform(&RNG) - 0.5f);
            }
            else {
                Scalar *= 2.;
                double* l = Array.ptr;
                double* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l += Scalar * (curand_uniform_double(&RNG) - 0.5);
            }
        }
#endif
        template <bool _MemoryOnHost, std::floating_point _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void RandomizeArray(Span<_T> Array, _T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
            if constexpr (_MemoryOnHost) {
                Scalar *= 2.f;
                std::uniform_real_distribution<_T> dis(-Scalar, Scalar);
                _T* l = Array.ptr;
                _T* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = std::clamp(*l + dis(RNG), LowerBound, UpperBound);
            }
            else {
                std::uniform_int_distribution<uint64_t> dis64(0);
                details::RandomizeArray_CallKernel(Array, Scalar, LowerBound, UpperBound, dis64(RNG));
            }
        }
#ifdef __CUDACC__
        template <std::floating_point _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void RandomizeArray(Span<_T> Array, _T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
            if constexpr (std::same_as<_T, float>) {
                Scalar *= 2.f;
                float* l = Array.ptr;
                float* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = std::clamp(*l + Scalar * (curand_uniform(&RNG) - 0.5f), LowerBound, UpperBound);
            }
            else {
                Scalar *= 2.f;
                double* l = Array.ptr;
                double* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = std::clamp(*l + Scalar * (curand_uniform_double(&RNG) - 0.5f), LowerBound, UpperBound);
            }
        }
#endif

        template <bool _MemoryOnHost, std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void RandomizeArrayWFlips(Span<_T> Array, uint32_t FlipProb, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
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
                    details::RandomizeArrayWFlips_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), FlipProb, dis64(RNG));

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
            else {
                RandomizeArrayWFlips<_MemoryOnHost, uint8_t, _TRNG>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), FlipProb, RNG);
            }
        }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void RandomizeArrayWFlips(Span<_T> Array, uint32_t FlipProb, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
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
            else {
                RandomizeArrayWFlips<uint8_t>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), FlipProb, RNG);
            }
        }
#endif
        template <bool _MemoryOnHost, std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void RandomizeArrayWTargets(Span<_T> Array, uint32_t EachFlipProb, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
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
                    details::RandomizeArrayWTargets_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), EachFlipProb, dis64(RNG));

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
            else {
                RandomizeArrayWTargets<_MemoryOnHost, uint8_t, _TRNG>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), EachFlipProb, RNG);
            }
        }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void RandomizeArrayWTargets(Span<_T> Array, uint32_t EachFlipProb, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
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
                RandomizeArrayWTargets<uint8_t>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), EachFlipProb, RNG);
            }
        }
#endif
        template <bool _MemoryOnHost, std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void RandomizeArrayWMutations(Span<_T> Array, uint32_t MutationProb, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
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
                    details::RandomizeArrayWMutations_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), MutationProb, dis64(RNG));

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
            else {
                RandomizeArrayWMutations<_MemoryOnHost, uint8_t, _TRNG>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), MutationProb, RNG);
            }
        }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void RandomizeArrayWMutations(Span<_T> Array, uint32_t MutationProb, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
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
                RandomizeArrayWMutations<uint8_t>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), MutationProb, RNG);
            }
        }
#endif
        template <bool _MemoryOnHost, std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void RandomizeArrayWMutations(Span<_T> Array, uint32_t MutationProb, uint32_t ProbabilityOf1, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
                if constexpr (_MemoryOnHost) {
                    uint64_t* l64 = (uint64_t*)Array.ptr;
                    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
                    for (; l64 < u64; ++l64) *l64 = RandomizeWMutations(*l64, MutationProb, ProbabilityOf1, RNG);

                    uint64_t endV;
                    size_t r = Array.size & 7;
                    memcpy(&endV, u64, r);
                    if (r > 4) endV = RandomizeWMutations(endV, MutationProb, ProbabilityOf1, RNG);
                    else if (r > 2) endV = (uint64_t)RandomizeWMutations((uint32_t)endV, MutationProb, ProbabilityOf1, RNG);
                    else if (r > 1) endV = (uint64_t)RandomizeWMutations((uint16_t)endV, MutationProb, ProbabilityOf1, RNG);
                    else endV = (uint64_t)RandomizeWMutations((uint8_t)endV, MutationProb, ProbabilityOf1, RNG);
                    memcpy(u64, &endV, r);
                }
                else {
                    std::uniform_int_distribution<uint64_t> dis64(0);
                    details::RandomizeArrayWMutations_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), MutationProb, ProbabilityOf1, dis64(RNG));

                    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

                    uint64_t endV;
                    size_t r = Array.size & 7;
                    cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
                    if (r > 4) endV = RandomizeWMutations(endV, MutationProb, ProbabilityOf1, RNG);
                    else if (r > 2) endV = (uint64_t)RandomizeWMutations((uint32_t)endV, MutationProb, ProbabilityOf1, RNG);
                    else if (r > 1) endV = (uint64_t)RandomizeWMutations((uint16_t)endV, MutationProb, ProbabilityOf1, RNG);
                    else endV = (uint64_t)RandomizeWMutations((uint8_t)endV, MutationProb, ProbabilityOf1, RNG);
                    cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);
                }
            }
            else {
                RandomizeArrayWMutations<_MemoryOnHost, uint8_t, _TRNG>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), MutationProb, ProbabilityOf1, RNG);
            }
        }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void RandomizeArrayWMutations(Span<_T> Array, uint32_t MutationProb, uint32_t ProbabilityOf1, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
                uint64_t* l64 = (uint64_t*)Array.ptr;
                uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
                for (; l64 < u64; ++l64) *l64 = RandomizeWMutations(*l64, MutationProb, ProbabilityOf1, RNG);

                uint64_t endV;
                size_t r = Array.size & 7;
                memcpy(&endV, u64, r);
                if (r > 4) endV = RandomizeWMutations(endV, MutationProb, ProbabilityOf1, RNG);
                else if (r > 2) endV = (uint64_t)RandomizeWMutations((uint32_t)endV, MutationProb, ProbabilityOf1, RNG);
                else if (r > 1) endV = (uint64_t)RandomizeWMutations((uint16_t)endV, MutationProb, ProbabilityOf1, RNG);
                else endV = (uint64_t)RandomizeWMutations((uint8_t)endV, MutationProb, ProbabilityOf1, RNG);
                memcpy(u64, &endV, r);
            }
            else {
                RandomizeArrayWMutations<uint8_t>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), MutationProb, ProbabilityOf1, RNG);
            }
        }
#endif

        template <bool _MemoryOnHost, typename _T, std::uniform_random_bit_generator _TRNG>
            requires std::is_arithmetic_v<_T>
        __host__ __forceinline void InitRandomArray(Span<_T> Array, _TRNG& RNG) {
            if constexpr (std::floating_point<_T>) {
                if constexpr (_MemoryOnHost) {
                    std::uniform_real_distribution<_T> dis(-1., 1.);
                    _T* l = Array.ptr;
                    _T* u = Array.ptr + Array.size;
                    for (; l < u; ++l) *l = dis(RNG);
                }
                else {
                    std::uniform_int_distribution<uint64_t> dis64(0);
                    details::InitArray_CallKernel(Array, dis64(RNG));
                }
            }
            else if (std::same_as<_T, uint8_t>) {
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
                }
                else {
                    std::uniform_int_distribution<uint64_t> dis64(0);
                    details::InitArray_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), dis64(RNG));

                    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

                    uint64_t endV;
                    size_t r = Array.size & 7;
                    cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
                    endV = dis64(RNG);
                    cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);
                }
            }
            else {
                InitRandomArray<_MemoryOnHost, uint8_t, _TRNG>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), RNG);
            }
        }
#ifdef __CUDACC__
        template <typename _T, bcuda::KernelCurandState _TRNG>
            requires std::is_arithmetic_v<_T>
        __device__ __forceinline void InitRandomArray(Span<_T> Array, _TRNG& RNG) {
            if constexpr (std::same_as<_T, float>) {
                float* l = Array.ptr;
                float* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = curand_uniform(&RNG);
            }
            else if constexpr (std::same_as<_T, double>) {
                double* l = Array.ptr;
                double* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = curand_uniform_double(&RNG);
            }
            else {
                static_assert(std::integral<_T>, "Type _T has not yet been implemented.");
                uint32_t* l32 = (uint32_t*)Array.ptr;
                uint32_t* u32 = ((uint32_t*)Array.ptr) + (Array.size >> 2);
                for (; l32 < u32; ++l32) *l32 = curand(&RNG);

                uint32_t endV;
                size_t r = Array.size & 3;
                memcpy(&endV, u32, r);
                endV = curand(&RNG);
                memcpy(u32, &endV, r);
            }
        }

#endif
        template <bool _MemoryOnHost, std::floating_point _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void InitRandomArray(Span<_T> Array, _T LowerBound, _T UpperBound, _TRNG& RNG) {
            if constexpr (_MemoryOnHost) {
                std::uniform_real_distribution<_T> dis(LowerBound, UpperBound);
                _T* l = Array.ptr;
                _T* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = dis(RNG);
            }
            else {
                std::uniform_int_distribution<uint64_t> dis64(0);
                details::InitArray_CallKernel(Array, LowerBound, UpperBound, dis64(RNG));
            }
        }
#ifdef __CUDACC__
        template <std::floating_point _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void InitRandomArray(Span<_T> Array, _T LowerBound, _T UpperBound, _TRNG& RNG) {
            if constexpr (std::same_as<_T, float>) {
                float range = UpperBound - LowerBound;
                float* l = Array.ptr;
                float* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = curand_uniform(&RNG) * range + LowerBound;
            }
            else {
                double range = UpperBound - LowerBound;
                double* l = Array.ptr;
                double* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = curand_uniform_double(&RNG) * range + LowerBound;
            }
        }
#endif

        template <bool _MemoryOnHost, std::integral _T, std::uniform_random_bit_generator _TRNG>
        __host__ __forceinline void InitRandomArray(Span<_T> Array, uint32_t ProbabilityOf1, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
                if constexpr (_MemoryOnHost) {
                    uint64_t* l64 = (uint64_t*)Array.ptr;
                    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
                    for (; l64 < u64; ++l64) *l64 = Get64Bits(ProbabilityOf1, RNG);

                    uint64_t endV;
                    size_t r = Array.size & 7;
                    memcpy(&endV, u64, r);
                    endV = Get64Bits(ProbabilityOf1, RNG);
                    memcpy(u64, &endV, r);
                }
                else {
                    std::uniform_int_distribution<uint64_t> dis64(0);
                    details::InitArray_CallKernel(Span<uint32_t>((uint32_t*)Array.ptr, Array.size >> 2), ProbabilityOf1, dis64(RNG));

                    uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);

                    uint64_t endV;
                    size_t r = Array.size & 7;
                    cudaMemcpy(&endV, u64, r, cudaMemcpyDefault);
                    endV = Get64Bits(ProbabilityOf1, RNG);
                    cudaMemcpy(u64, &endV, r, cudaMemcpyDefault);
                }
            }
            else {
                InitRandomArray<_MemoryOnHost, uint8_t, _TRNG>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), ProbabilityOf1, RNG);
            }
        }
#ifdef __CUDACC__
        template <std::integral _T, bcuda::KernelCurandState _TRNG>
        __device__ __forceinline void InitRandomArray(Span<_T> Array, uint32_t ProbabilityOf1, _TRNG& RNG) {
            if constexpr (std::same_as<_T, uint8_t>) {
                uint32_t* l32 = (uint32_t*)Array.ptr;
                uint32_t* u32 = ((uint32_t*)Array.ptr) + (Array.size >> 2);
                for (; l32 < u32; ++l32) *l32 = Get32Bits(ProbabilityOf1, RNG);

                uint32_t endV;
                size_t r = Array.size & 3;
                memcpy(&endV, u32, r);
                endV = Get32Bits(ProbabilityOf1, RNG);
                memcpy(u32, &endV, r);
            }
            else {
                InitRandomArray<uint8_t>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)), ProbabilityOf1, RNG);
            }
        }
#endif

        template <bool _MemoryOnHost, typename _T>
            requires std::is_arithmetic_v<_T>
        __host__ __forceinline void ClearArray(Span<_T> Array) {
            if constexpr (std::floating_point<_T>) {
                if constexpr (_MemoryOnHost) {
                    _T* l = Array.ptr;
                    _T* u = Array.ptr + Array.size;
                    for (; l < u; ++l) *l = (_T)0;
                }
                else {
                    std::uniform_int_distribution<uint64_t> dis64(0);
                    details::ClearArray_CallKernel(Array);
                }
            }
            else if constexpr (std::same_as<_T, uint8_t>) {
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
            else {
                ClearArray<_MemoryOnHost, uint8_t>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)));
            }
        }
#ifdef __CUDACC__
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        __device__ __forceinline void ClearArray(Span<_T> Array) {
            if constexpr (std::floating_point<_T>) {
                _T* l = Array.ptr;
                _T* u = Array.ptr + Array.size;
                for (; l < u; ++l) *l = (_T)0;
            }
            else if constexpr (std::same_as<_T, uint8_t>) {
                uint64_t* l64 = (uint64_t*)Array.ptr;
                uint64_t* u64 = ((uint64_t*)Array.ptr) + (Array.size >> 3);
                for (; l64 < u64; ++l64) *l64 = 0;

                uint64_t endV = 0;
                size_t r = Array.size & 7;
                memcpy(u64, &endV, r);
            }
            else {
                ClearArray<uint8_t>(Span<uint8_t>((uint8_t*)Array.ptr, Array.size * sizeof(_T)));
            }
        }
#endif
    }
}