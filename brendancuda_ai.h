#pragma once

#include "brendancuda_arrays.h"
#include "brendancuda_rand_bits.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

namespace BrendanCUDA {
    namespace AI {
        template <typename _T>
        using activationFunction_t = _T(*)(_T Value);

        template <typename _T>
        __host__ __device__ constexpr _T ReLU(_T Value);
        template <typename _T, _T _Slope>
        __host__ __device__ constexpr _T LeakyReLU(_T Value);
        template <typename _T, _T _Lower, _T _Upper>
        __host__ __device__ constexpr _T BoundReLU(_T Value);
        template <typename _T, _T _Slope, _T _Lower, _T _Upper>
        __host__ __device__ constexpr _T LeakyBoundReLU(_T Value);
        template <typename _T>
        __host__ __device__ _T TanH(_T Value);
        template <typename _T>
        __host__ __device__ _T Sigmoid(_T Value);

        template <bool _FloatsOnHost, std::floating_point _TFloat, bool _BoolsOnHost>
        __host__ void ConvertFloatsToBools(_TFloat* Floats, bool* Bools, size_t Length, _TFloat Split);
#ifdef __CUDACC__
        template <std::floating_point _TFloat>
        __device__ void ConvertFloatsToBools(_TFloat* Floats, bool* Bools, size_t Length, _TFloat Split);
#endif
        template <bool _FloatsOnHost, std::floating_point _TFloat, bool _IntsOnHost, std::integral _TInt>
        __host__ void ConvertFloatsToInts(_TFloat* Floats, _TInt* Ints, size_t FloatsLength, _TFloat Split);
#ifdef __CUDACC__
        template <std::floating_point _TFloat, std::integral _TInt>
        __device__ void ConvertFloatsToInts(_TFloat* Floats, _TInt* Ints, size_t FloatsLength, _TFloat Split);
#endif

        template <bool _BoolsOnHost, bool _FloatsOnHost, std::floating_point _TFloat>
        __host__ void ConvertBoolsToFloats(bool* Bools, _TFloat* Floats, size_t Length, _TFloat ValFalse, _TFloat ValTrue);
#ifdef __CUDACC__
        template <std::floating_point _TFloat>
        __device__ void ConvertBoolsToFloats(bool* Bools, _TFloat* Floats, size_t Length, _TFloat ValFalse, _TFloat ValTrue);
#endif
        template <bool _IntsOnHost, std::integral _TInt, bool _FloatsOnHost, std::floating_point _TFloat>
        __host__ void ConvertIntsToFloats(_TInt* Ints, _TFloat* Floats, size_t FloatsLength, _TFloat ValFalse, _TFloat ValTrue);
#ifdef __CUDACC__
        template <std::integral _TInt, std::floating_point _TFloat>
        __device__ void ConvertIntsToFloats(_TInt* Ints, _TFloat* Floats, size_t FloatsLength, _TFloat ValFalse, _TFloat ValTrue);
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