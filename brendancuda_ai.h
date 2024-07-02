#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "brendancuda_random_bits.h"
#include "brendancuda_random_anyrng.h"
#include <limits>
#include "brendancuda_arrays.h"

namespace BrendanCUDA {
    namespace AI {
        //An activation function, typically used for an MLP.
        template <typename _T>
        using activationFunction_t = _T(*)(_T Value);

        //The ReLU activation function, compatible with activationFunction_t.
        template <typename _T>
        __host__ __device__ _T ReLU(_T Value);
        //The Leaky ReLU activation function, compatible with activationFunction_t.
        template <typename _T, _T _Slope>
        __host__ __device__ _T LeakyReLU(_T Value);
        //The ReLU activation function, not maximized, but clamped, and compatible with activationFunction_t.
        template <typename _T, _T _Lower, _T _Upper>
        __host__ __device__ _T BoundReLU(_T Value);
        //The Leaky ReLU activation function, not maximized, but clamped, and compatible with activationFunction_t.
        template <typename _T, _T _Slope, _T _Lower, _T _Upper>
        __host__ __device__ _T LeakyBoundReLU(_T Value);
        //The TanH activation function, compatible with activationFunction_t.
        template <typename _T>
        __host__ __device__ _T TanH(_T Value);
        //The Sigmoid activation function, like the TanH, but worse, though still compatible with activationFunction_t.
        template <typename _T>
        __host__ __device__ _T Sigmoid(_T Value);

        //Makes a small, random adjustment in the range of [-Scalar, +Scalar] to each of the values of Array (Array being a pointer to VRAM).
        __host__ __device__ void RandomizeArray(Span<float> Array, float Scalar, Random::AnyRNG<uint64_t> RNG);
        //Makes a small, random adjustment in the range of [-Scalar, +Scalar] to each of the values of Array (Array being a pointer to VRAM), clamping the final value in the range of [LowerBound, UpperBound], even if it wasn't in that range to start.
        __host__ __device__ void RandomizeArray(Span<float> Array, float Scalar, float LowerBound, float UpperBound, Random::AnyRNG<uint64_t> RNG);

        //Makes a small, random adjustment in the range of [-Scalar, +Scalar] to each of the values of Array (Array being a pointer to VRAM).
        __host__ __device__ void RandomizeArray(Span<double> Array, double Scalar, Random::AnyRNG<uint64_t> RNG);
        //Makes a small, random adjustment in the range of [-Scalar, +Scalar] to each of the values of Array (Array being a pointer to VRAM), clamping the final value in the range of [LowerBound, UpperBound], even if it wasn't in that range to start.
        __host__ __device__ void RandomizeArray(Span<double> Array, double Scalar, double LowerBound, double UpperBound, Random::AnyRNG<uint64_t> RNG);

        //Flips bits in Array (Array being a pointer to VRAM), the probability of each bit being flipped a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void RandomizeArray(Span<uint64_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);
        //Flips bits in Array (Array being a pointer to VRAM), the probability of each bit being flipped a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void RandomizeArray(Span<uint32_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);
        //Flips bits in Array (Array being a pointer to VRAM), the probability of each bit being flipped a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void RandomizeArray(Span<uint16_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);
        //Flips bits in Array (Array being a pointer to VRAM), the probability of each bit being flipped a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void RandomizeArray(Span<uint8_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);

        //Fills Array (Array being a pointer to VRAM) with random numbers uniformly distributed in the range of [0, 1].
        __host__ __device__ void InitRandomArray(Span<float> Array, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random numbers uniformly distributed in the range of [LowerBound, UpperBound].
        __host__ __device__ void InitRandomArray(Span<float> Array, float LowerBound, float UpperBound, Random::AnyRNG<uint64_t> RNG);

        //Fills Array (Array being a pointer to VRAM) with random numbers uniformly distributed in the range of [0, 1].
        __host__ __device__ void InitRandomArray(Span<double> Array, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random numbers uniformly distributed in the range of [LowerBound, UpperBound].
        __host__ __device__ void InitRandomArray(Span<double> Array, double LowerBound, double UpperBound, Random::AnyRNG<uint64_t> RNG);

        //Fills Array (Array being a pointer to VRAM) with random bits.
        __host__ __device__ void InitRandomArray(Span<uint64_t> Array, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random bits.
        __host__ __device__ void InitRandomArray(Span<uint32_t> Array, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random bits.
        __host__ __device__ void InitRandomArray(Span<uint16_t> Array, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random bits.
        __host__ __device__ void InitRandomArray(Span<uint8_t> Array, Random::AnyRNG<uint64_t> RNG);

        //Fills Array (Array being a pointer to VRAM) with random bits, each bit having their probability of being 1 as a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void InitRandomArray(Span<uint64_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random bits, each bit having their probability of being 1 as a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void InitRandomArray(Span<uint32_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random bits, each bit having their probability of being 1 as a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void InitRandomArray(Span<uint16_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);
        //Fills Array (Array being a pointer to VRAM) with random bits, each bit having their probability of being 1 as a function of ProbabilityOf1, where the binary digits of ProbabilityOf1 are binary digits after the decimal point of a base-2 number.
        __host__ __device__ void InitRandomArray(Span<uint8_t> Array, uint32_t ProbabilityOf1, Random::AnyRNG<uint64_t> RNG);

        //Fills Array (Array being a pointer to VRAM) with all 0s.
        __host__ __device__ void InitZeroArray(Span<float> Array);
        //Fills Array (Array being a pointer to VRAM) with all 0s.
        __host__ __device__ void InitZeroArray(Span<double> Array);
        //Fills Array (Array being a pointer to VRAM) with all 0s.
        __host__ __device__ void InitZeroArray(Span<uint64_t> Array);
        //Fills Array (Array being a pointer to VRAM) with all 0s.
        __host__ __device__ void InitZeroArray(Span<uint32_t> Array);
        //Fills Array (Array being a pointer to VRAM) with all 0s.
        __host__ __device__ void InitZeroArray(Span<uint16_t> Array);
        //Fills Array (Array being a pointer to VRAM) with all 0s.
        __host__ __device__ void InitZeroArray(Span<uint8_t> Array);

        //Converts an array of floats (Floats) to bools, writing the result to Bools. Each bool is true if its corresponding float is greater than Split, and false otherwise. MemoryOnHost specifies whether or not the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split, bool MemoryOnHost);
        //Converts an array of doubles (Doubles) to bools, writing the result to Bools. Each bool is true if its corresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or not the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyDoublesToBools(float* Doubles, bool* Bools, size_t Length, float Split, bool MemoryOnHost);
        //Converts an array of floats (Floats) to bools, writing the result to Bools. Each bool is true if its corresponding float is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split);
        //Converts an array of doubles (Doubles) to bools, writing the result to Bools. Each bool is true if its corresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyDoublesToBools(double* Doubles, bool* Bools, size_t Length, double Split);

        //Converts an array of floats (Floats) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many floats as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split, bool MemoryOnHost);
        //Converts an array of doubles (Doubles) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many doubles as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split, bool MemoryOnHost);
        //Converts an array of floats (Floats) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many floats as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split);
        //Converts an array of doubles (Doubles) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many doubles as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split);
        //Converts an array of floats (Floats) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many floats as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split, bool MemoryOnHost);
        //Converts an array of doubles (Doubles) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many doubles as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split, bool MemoryOnHost);
        //Converts an array of floats (Floats) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many floats as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split);
        //Converts an array of doubles (Doubles) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many doubles as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split);
    }
}

template <typename _T, _T _Slope>
__host__ __device__ _T BrendanCUDA::AI::LeakyReLU(_T Value) {
    return (Value < (_T)0.) ? Value * _Slope : Value;
}
template <typename _T, _T _Lower, _T _Upper>
__host__ __device__ _T BrendanCUDA::AI::BoundReLU(_T Value) {
    if (Value < _Lower) return (_T)0.;
    if (Value > _Upper) return (_T)1.;
    return Value;
}
template <typename _T, _T _Slope, _T _Lower, _T _Upper>
__host__ __device__ _T BrendanCUDA::AI::LeakyBoundReLU(_T Value) {
    if (Value < _Lower) return Value * _Slope;
    if (Value > _Upper) return (_T)1. + (Value - _Upper) * _Slope;
    return Value;
}