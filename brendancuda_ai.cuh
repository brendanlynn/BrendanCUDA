#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "brendancuda_random_devicerandom.cuh"
#include "brendancuda_random_sseed.cuh"
#include "brendancuda_random_bits.cuh"
#include "brendancuda_random_rngfunc.cuh"
#include "brendancuda_macros.cuh"
#include <limits>

namespace BrendanCUDA {
    namespace AI {
        template <typename T>
        using activationFunction_t = T(*)(T);

        __global__ void RandomizeArrayKernel(float* Array, float Scalar, uint64_t Seed);
        __host__ __device__ void RandomizeArray(float* Array, size_t Length, float Scalar, Random::rngWState<uint64_t> rng);
        __global__ void RandomizeArrayKernel(float* Array, float Scalar, float LowerBound, float UpperBound, uint64_t Seed);
        __host__ __device__ void RandomizeArray(float* Array, size_t Length, float Scalar, float LowerBound, float UpperBound, Random::rngWState<uint64_t> rng);

        __global__ void RandomizeArrayKernel(double* Array, double Scalar, uint64_t Seed);
        __host__ __device__ void RandomizeArray(double* Array, size_t Length, double Scalar, Random::rngWState<uint64_t> rng);
        __global__ void RandomizeArrayKernel(double* Array, double Scalar, double LowerBound, double UpperBound, uint64_t Seed);
        __host__ __device__ void RandomizeArray(double* Array, size_t Length, double Scalar, double LowerBound, double UpperBound, Random::rngWState<uint64_t> rng);

        __global__ void RandomizeArrayKernel(uint64_t* Array, uint32_t ProbabilityOf1, uint64_t Seed);
        __host__ __device__ void RandomizeArray(uint64_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);
        __host__ __device__ void RandomizeArray(uint32_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);
        __host__ __device__ void RandomizeArray(uint16_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);
        __host__ __device__ void RandomizeArray(uint8_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);

        __global__ void InitRandomArrayKernel(float* Array, uint64_t Seed);
        __host__ __device__ void InitRandomArray(float* Array, size_t Length, Random::rngWState<uint64_t> rng);
        __global__ void InitRandomArrayKernel(float* Array, float LowerBound, float UpperBound, uint64_t Seed);
        __host__ __device__ void InitRandomArray(float* Array, size_t Length, float LowerBound, float UpperBound, Random::rngWState<uint64_t> rng);

        __global__ void InitRandomArrayKernel(double* Array, uint64_t Seed);
        __host__ __device__ void InitRandomArray(double* Array, size_t Length, Random::rngWState<uint64_t> rng);
        __global__ void InitRandomArrayKernel(double* Array, double LowerBound, double UpperBound, uint64_t Seed);
        __host__ __device__ void InitRandomArray(double* Array, size_t Length, double LowerBound, double UpperBound, Random::rngWState<uint64_t> rng);

        __global__ void InitRandomArrayKernel(uint64_t* Array, uint64_t Seed);
        __host__ __device__ void InitRandomArray(uint64_t* Array, size_t Length, Random::rngWState<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint32_t* Array, size_t Length, Random::rngWState<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint16_t* Array, size_t Length, Random::rngWState<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint8_t* Array, size_t Length, Random::rngWState<uint64_t> rng);

        __global__ void InitRandomArrayKernel(uint64_t* Array, uint32_t ProbabilityOf1, uint64_t Seed);
        __host__ __device__ void InitRandomArray(uint64_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint32_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint16_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);
        __host__ __device__ void InitRandomArray(uint8_t* Array, size_t Length, uint32_t ProbabilityOf1, Random::rngWState<uint64_t> rng);

        __global__ void InitZeroArrayKernel(float* Array);
        __host__ __device__ void InitZeroArray(float* Array, size_t Length);
        __global__ void InitZeroArrayKernel(double* Array);
        __host__ __device__ void InitZeroArray(double* Array, size_t Length);
        __global__ void InitZeroArrayKernel(uint64_t* Array);
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

        __host__ __device__ void CopyFloatsToInt32Func(float* Floats, uint32_t* Int32, float Split);
        __host__ __device__ void CopyDoublesToInt32Func(double* Floats, uint32_t* Int32, double Split);
        __host__ __device__ void CopyFloatsToInt64Func(float* Floats, uint64_t* Int64, float Split);
        __host__ __device__ void CopyDoublesToInt64Func(double* Floats, uint64_t* Int64, double Split);

        __global__ void CopyFloatsToInt32sKernel(float* Floats, uint32_t* Int32s, float Split);
        __host__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split, bool MemoryOnHost);
        __global__ void CopyDoublesToInt32sKernel(double* Doubles, uint32_t* Int32s, double Split);
        __host__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split, bool MemoryOnHost);
        __device__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split);
        __device__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split);
        __global__ void CopyFloatsToInt64sKernel(float* Floats, uint64_t* Int64s, float Split);
        __host__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split, bool MemoryOnHost);
        __global__ void CopyDoublesToInt64sKernel(double* Doubles, uint64_t* Int64s, double Split);
        __host__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split, bool MemoryOnHost);
        __device__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split);
        __device__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split);
    }
}