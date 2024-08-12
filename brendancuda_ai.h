#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "brendancuda_rand_bits.h"
#include "brendancuda_rand_anyrng.h"
#include <limits>
#include "brendancuda_arrays.h"

namespace BrendanCUDA {
    namespace AI {
        //An activation function, typically used for an MLP.
        template <typename _T>
        using activationFunction_t = _T(*)(_T Value);

        //The ReLU activation function, compatible with activationFunction_t.
        template <typename _T>
        __host__ __device__ constexpr _T ReLU(_T Value);
        //The Leaky ReLU activation function, compatible with activationFunction_t.
        template <typename _T, _T _Slope>
        __host__ __device__ constexpr _T LeakyReLU(_T Value);
        //The ReLU activation function, not maximized, but clamped, and compatible with activationFunction_t.
        template <typename _T, _T _Lower, _T _Upper>
        __host__ __device__ constexpr _T BoundReLU(_T Value);
        //The Leaky ReLU activation function, not maximized, but clamped, and compatible with activationFunction_t.
        template <typename _T, _T _Slope, _T _Lower, _T _Upper>
        __host__ __device__ constexpr _T LeakyBoundReLU(_T Value);
        //The TanH activation function, compatible with activationFunction_t.
        template <typename _T>
        __host__ __device__ _T TanH(_T Value);
        //The Sigmoid activation function, like the TanH, but worse, though still compatible with activationFunction_t.
        template <typename _T>
        __host__ __device__ _T Sigmoid(_T Value);

        //Converts an array of floats (Floats) to bools, writing the result to Bools. Each bool is true if its corresponding float is greater than Split, and false otherwise. MemoryOnHost specifies whether or not the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split, bool MemoryOnHost);
        //Converts an array of doubles (Doubles) to bools, writing the result to Bools. Each bool is true if its corresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or not the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyDoublesToBools(float* Doubles, bool* Bools, size_t Length, float Split, bool MemoryOnHost);
#ifdef __CUDACC__
        //Converts an array of floats (Floats) to bools, writing the result to Bools. Each bool is true if its corresponding float is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split);
        //Converts an array of doubles (Doubles) to bools, writing the result to Bools. Each bool is true if its corresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyDoublesToBools(double* Doubles, bool* Bools, size_t Length, double Split);
#endif

        //Converts an array of floats (Floats) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many floats as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split, bool MemoryOnHost);
        //Converts an array of doubles (Doubles) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many doubles as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split, bool MemoryOnHost);
#ifdef __CUDACC__
        //Converts an array of floats (Floats) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many floats as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split);
        //Converts an array of doubles (Doubles) to an array of 32-bit unsigned integers (Int32s). There should be 32 times as many doubles as 32-bit unsigned integers, with Int32Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split);
#endif
        //Converts an array of floats (Floats) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many floats as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split, bool MemoryOnHost);
        //Converts an array of doubles (Doubles) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many doubles as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. MemoryOnHost specifies whether or the pointers point to memory on the RAM or the VRAM (true being the RAM, false being the VRAM).
        __host__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split, bool MemoryOnHost);
#ifdef __CUDACC__
        //Converts an array of floats (Floats) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many floats as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split);
        //Converts an array of doubles (Doubles) to an array of 64-bit unsigned integers (Int64s). There should be 64 times as many doubles as 64-bit unsigned integers, with Int64Length representing the quantity of the integers. Each bit is true if its cooresponding double is greater than Split, and false otherwise. The pointers are expected to point to memory on the VRAM.
        __device__ void CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split);
#endif
    }
}

template <typename _T>
__host__ __device__ constexpr _T BrendanCUDA::AI::ReLU(_T Value) {
    return (Value < (_T)0.) ? (_T)0. : Value;
}
template <typename _T, _T _Slope>
__host__ __device__ constexpr _T BrendanCUDA::AI::LeakyReLU(_T Value) {
    return (Value < (_T)0.) ? Value * _Slope : Value;
}
template <typename _T, _T _Lower, _T _Upper>
__host__ __device__ constexpr _T BrendanCUDA::AI::BoundReLU(_T Value) {
    if (Value < _Lower) return (_T)0.;
    if (Value > _Upper) return (_T)1.;
    return Value;
}
template <typename _T, _T _Slope, _T _Lower, _T _Upper>
__host__ __device__ constexpr _T BrendanCUDA::AI::LeakyBoundReLU(_T Value) {
    if (Value < _Lower) return Value * _Slope;
    if (Value > _Upper) return (_T)1. + (Value - _Upper) * _Slope;
    return Value;
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::AI::TanH(_T Value) {
    return std::tanh(Value);
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::AI::Sigmoid(_T Value) {
    Value = std::exp(Value);
    return Value / ((_T)1. + Value);
}