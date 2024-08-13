#pragma once

#include "brendancuda_ai.h"
#include "brendancuda_dcopy.cuh"
#include "brendancuda_errorhelp.h"
#include "brendancuda_rand_randomizer.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>
#include <random>

namespace BrendanCUDA {
    namespace AI {
        namespace MLP {
            template <typename _T>
            class MLPL {
                static_assert(std::is_same<_T, float>::value || std::is_same<_T, double>::value, "_T must be either 'float' or 'double'.");
            public:
                __forceinline MLPL() = default;
                __host__ __device__ __forceinline MLPL(size_t InputLength, size_t OutputLength);
                __host__ __device__ __forceinline MLPL(size_t InputLength, size_t OutputLength, _T* Weights, _T* Bias);

                __host__ __device__ __forceinline void Dispose();

                __host__ __device__ __forceinline void ZeroOverwrite();
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline void RandomOverwrite(_TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline void RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG);

                __host__ __device__ __forceinline size_t InputLength() const;
                __host__ __device__ __forceinline size_t OutputLength() const;
                __host__ __device__ __forceinline _T* Weights() const;
                __host__ __device__ __forceinline _T* Weight(size_t Index) const;
                __host__ __device__ __forceinline _T* Weight(size_t X, size_t Y) const;
                __host__ __device__ __forceinline _T* Bias() const;
                __host__ __device__ __forceinline _T* Bias(size_t Index) const;

                template <bool _CopyToHost>
                __host__ __forceinline _T* GetWeights() const;
#ifdef __CUDACC__
                __device__ __forceinline _T* GetWeights() const;
#endif
                __host__ __device__ __forceinline _T GetWeight(size_t Index);
                __host__ __device__ __forceinline _T GetWeight(size_t X, size_t Y);
                __host__ __device__ __forceinline void SetWeights(_T* Values);
                __host__ __device__ __forceinline void SetWeight(_T Value, size_t Index);
                __host__ __device__ __forceinline void SetWeight(_T Value, size_t X, size_t Y);
                template <bool _CopyToHost>
                __host__ __forceinline _T* GetBias() const;
#ifdef __CUDACC__
                __device__ __forceinline _T* GetBias() const;
#endif
                __host__ __device__ __forceinline _T GetBias(size_t Index) const;
                __host__ __device__ __forceinline void SetBias(_T* Values);
                __host__ __device__ __forceinline void SetBias(_T Value, size_t Index);

                __host__ __forceinline _T* Run(_T* Input) const;

                __host__ __device__ __forceinline MLPL<_T> Clone() const;
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline void Randomize(_T Scalar, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline void Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline MLPL<_T> Reproduce(_T Scalar, _TRNG& RNG) const;
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline MLPL<_T> Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) const;
            private:
                size_t iptLen;
                size_t optLen;
                _T* wgts;
                _T* bias;
            };
        }
    }
}

template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T>::MLPL(size_t InputLength, size_t OutputLength) {
    iptLen = InputLength;
    optLen = OutputLength;
#ifdef __CUDA_ARCH__
    wgts = (_T*)operator new[](InputLength* OutputLength * sizeof(_T));
    bias = (_T*)operator new[](OutputLength * sizeof(_T));
#else
    ThrowIfBad(cudaMalloc(&wgts, InputLength * OutputLength * sizeof(_T)));
    ThrowIfBad(cudaMalloc(&bias, OutputLength * sizeof(_T)));
#endif
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T>::MLPL(size_t InputLength, size_t OutputLength, _T* Weights, _T* Bias)
    : MLPL(InputLength, OutputLength) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(wgts, Weights, InputLength * OutputLength * sizeof(_T));
    deviceMemcpy(bias, Bias, OutputLength * sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(wgts, Weights, InputLength * OutputLength * sizeof(_T), cudaMemcpyDefault));
    ThrowIfBad(cudaMemcpy(bias, Bias, OutputLength * sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::ZeroOverwrite() {
    Random::ClearArray<false, _T>(Span<_T>(wgts, iptLen * optLen));
    Random::ClearArray<false, _T>(Span<_T>(bias, optLen));
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::RandomOverwrite(_TRNG& RNG) {
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(wgts, iptLen * optLen), RNG);
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(bias, optLen), RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(wgts, iptLen * optLen), LowerBound, UpperBound, RNG);
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(bias, optLen), LowerBound, UpperBound, RNG);
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::AI::MLP::MLPL<_T>::InputLength() const {
    return iptLen;
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::AI::MLP::MLPL<_T>::OutputLength() const {
    return optLen;
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::Weights() const {
    return wgts;
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::Bias() const {
    return bias;
}
template <typename _T>
__host__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::Run(_T* Input) const {
    _T* output;
    ThrowIfBad(cudaMalloc(&output, sizeof(_T) * optLen));
    ThrowIfBad(cudaMemcpy(output, bias, sizeof(float) * optLen, cudaMemcpyDeviceToDevice));

    cublasHandle_t h;
    cublasCreate(&h);

    if constexpr (std::is_same<float, _T>::value) {
        float f = 1.0f;
        cublasSgemv(h, CUBLAS_OP_N, optLen, iptLen, &f, (float*)wgts, optLen, (float*)Input, 1, &f, (float*)output, 1);
    }
    else {
        double d = 1.0;
        cublasDgemv(h, CUBLAS_OP_N, optLen, iptLen, &d, (double*)wgts, optLen, (double*)Input, 1, &d, (double*)output, 1);
    }

    cublasDestroy(h);

    return output;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::Dispose() {
#ifdef __CUDA_ARCH__
    delete[] wgts;
    delete[] bias;
#else
    ThrowIfBad(cudaFree(wgts));
    ThrowIfBad(cudaFree(bias));
#endif
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::Weight(size_t Index) const {
    return &wgts[Index];
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::Weight(size_t X, size_t Y) const {
    return &wgts[X * optLen + Y];
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::Bias(size_t Index) const {
    return &bias[Index];
}
template <typename _T>
template <bool _CopyToHost>
__host__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetWeights() const {
    size_t l = iptLen * optLen * sizeof(_T);
    if constexpr (_CopyToHost) {
        _T* output = (_T*)operator new[](l);
        ThrowIfBad(cudaMemcpy(output, wgts, l, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        _T* output;
        ThrowIfBad(cudaMalloc(&output, l));
        ThrowIfBad(cudaMemcpy(output, wgts, l, cudaMemcpyDeviceToHost));
        return output;
    }
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetWeights() const {
    size_t l = iptLen * optLen * sizeof(_T);
    _T* output = (_T*)operator new[](l);
    deviceMemcpy(output, wgts, l);
    return output;
}
#endif
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::AI::MLP::MLPL<_T>::GetWeight(size_t Index) {
#ifdef __CUDA_ARCH__
    return wgts[Index];
#else
    _T v;
    ThrowIfBad(cudaMemcpy(&v, Weight(Index), sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::AI::MLP::MLPL<_T>::GetWeight(size_t X, size_t Y) {
#ifdef __CUDA_ARCH__
    return *Weight(X, Y);
#else
    _T v;
    ThrowIfBad(cudaMemcpy(&v, Weight(X, Y), sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::SetWeights(_T* Values) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(wgts, Values, iptLen * optLen * sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(wgts, Values, iptLen * optLen * sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::SetWeight(_T Value, size_t Index) {
#ifdef __CUDA_ARCH__
    wgts[Index] = Value;
#else
    ThrowIfBad(cudaMemcpy(Weight(Index), &Value, sizeof(_T), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::SetWeight(_T Value, size_t X, size_t Y) {
#ifdef __CUDA_ARCH__
    * Weight(X, Y) = Value;
#else
    ThrowIfBad(cudaMemcpy(Weight(X, Y), &Value, sizeof(_T), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
template <bool _CopyToHost>
__host__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetBias() const {
    size_t l = optLen * sizeof(_T);
    if constexpr (_CopyToHost) {
        _T* output = (_T*)operator new[](l);
        ThrowIfBad(cudaMemcpy(output, bias, l, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        _T* output;
        ThrowIfBad(cudaMalloc(&output, l));
        ThrowIfBad(cudaMemcpy(output, bias, l, cudaMemcpyDeviceToHost));
        return output;
    }
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetBias() const {
    size_t l = optLen * sizeof(_T);
    _T* output = (_T*)operator new[](l);
    deviceMemcpy(output, bias, l);
    return output;
}
#endif
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::AI::MLP::MLPL<_T>::GetBias(size_t Index) const {
#ifdef __CUDA_ARCH__
    return bias[Index];
#else
    _T v;
    ThrowIfBad(cudaMemcpy(&v, Bias(Index), sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::SetBias(_T* Values) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(bias, Values, optLen * sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(bias, Values, optLen * sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::SetBias(_T Value, size_t Index) {
#ifdef __CUDA_ARCH__
    bias[Index] = Value;
#else
    ThrowIfBad(cudaMemcpy(Bias(Index), &Value, sizeof(_T), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLPL<_T>::Clone() const {
    return MLPL<_T>(iptLen, optLen, wgts, bias);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::Randomize(_T Scalar, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(wgts, iptLen * optLen), Scalar, RNG);
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(bias, optLen), Scalar, RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLPL<_T>::Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(wgts, iptLen * optLen), Scalar, LowerBound, UpperBound, RNG);
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(bias, optLen), Scalar, LowerBound, UpperBound, RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLPL<_T>::Reproduce(_T Scalar, _TRNG& RNG) const {
    MLPL<_T> n = Clone();
    n.Randomize(Scalar, RNG);
    return n;
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLPL<_T>::Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) const {
    MLPL<_T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, RNG);
    return n;
}