#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "brendancuda_random_devicerandom.cuh"
#include "brendancuda_random_sseed.cuh"
#include "brendancuda_random_bits.cuh"
#include "brendancuda_random_anyrng.cuh"
#include <limits>

namespace BrendanCUDA {
    namespace AI {
        template <typename _T>
        using activationFunction_t = _T(*)(_T Value);

        __host__ __device__ void RandomizeArray(float* Array, size_t Length, float Scalar, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void RandomizeArray(float* Array, size_t Length, float Scalar, float LowerBound, float UpperBound, Random::AnyRNG<uint64_t> rng);

        __host__ __device__ void RandomizeArray(double* Array, size_t Length, double Scalar, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void RandomizeArray(double* Array, size_t Length, double Scalar, double LowerBound, double UpperBound, Random::AnyRNG<uint64_t> rng);

        __host__ __device__ void RandomizeArray(uint64_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void RandomizeArray(uint32_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void RandomizeArray(uint16_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void RandomizeArray(uint8_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);

        __host__ __device__ void InitRandomArray(float* Array, size_t Length, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(float* Array, size_t Length, float LowerBound, float UpperBound, Random::AnyRNG<uint64_t> rng);

        __host__ __device__ void InitRandomArray(double* Array, size_t Length, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(double* Array, size_t Length, double LowerBound, double UpperBound, Random::AnyRNG<uint64_t> rng);

        __host__ __device__ void InitRandomArray(uint64_t* Array, size_t Length, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint32_t* Array, size_t Length, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint16_t* Array, size_t Length, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint8_t* Array, size_t Length, Random::AnyRNG<uint64_t> rng);

        __host__ __device__ void InitRandomArray(uint64_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint32_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint16_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint8_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> rng);

        __host__ __device__ void InitZeroArray(float* Array, size_t Length);
        __host__ __device__ void InitZeroArray(double* Array, size_t Length);
        __host__ __device__ void InitZeroArray(uint64_t* Array, size_t Length);
        __host__ __device__ void InitZeroArray(uint32_t* Array, size_t Length);
        __host__ __device__ void InitZeroArray(uint16_t* Array, size_t Length);
        __host__ __device__ void InitZeroArray(uint8_t* Array, size_t Length);

        __global__ void CopyFloatsToBoolsKernel(float* Floats, bool* Bools, float Split);
        __host__ void CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split, bool MemoryOnHost);

        __global__ void CopyDoublesToBoolsKernel(float* Doubles, bool* Bools, float Split);
        __host__ void CopyDoublesToBools(float* Doubles, bool* Bools, size_t Length, float Split, bool MemoryOnHost);

        __device__ void CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split);

        __device__ void CopyDoublesToBools(double* Doubles, bool* Bools, size_t Length, double Split);

        __host__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split, bool MemoryOnHost);
        __host__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split, bool MemoryOnHost);
        __device__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split);
        __device__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split);
        __host__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split, bool MemoryOnHost);
        __host__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split, bool MemoryOnHost);
        __device__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split);
        __device__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split);
    }
}