#pragma once

#include "brendancuda_ai_mlp_mlpl.h"
#include "brendancuda_arrays.h"
#include "brendancuda_errorhelp.h"
#include "brendancuda_rand_anyrng.h"

namespace BrendanCUDA {
    namespace details {
        template <typename _T>
        __host__ __device__ void RunActivationFunctionOnArray(Span<_T> Array, AI::activationFunction_t<_T> ActivationFunction);
    }
    namespace AI {
        namespace MLP {
            template <typename _T>
            class MLP final {
                static_assert(std::is_same<_T, float>::value || std::is_same<_T, double>::value, "_T must be either 'float' or 'double'.");
            public:
                __forceinline MLP() = default;
                __host__ __device__ __forceinline MLP(size_t Length, activationFunction_t<_T> ActivationFunction);
                __host__ __device__ __forceinline MLP(size_t Length, activationFunction_t<_T> ActivationFunction, MLPL<_T>* Layers);

                __host__ __device__ __forceinline MLPL<_T>* Layers() const;
                __host__ __device__ __forceinline size_t LayerCount() const;
                __host__ __device__ __forceinline activationFunction_t<_T> ActivationFunction() const;

                __host__ __device__ __forceinline void Dispose();

                __host__ __device__ __forceinline void ZeroOverwrite();
                __host__ __device__ __forceinline void RandomOverwrite(Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);

                __host__ __device__ __forceinline MLPL<_T>* Layer(size_t LayerIndex) const;
                template <bool _CopyToHost>
                __host__ __forceinline MLPL<_T>* GetLayers() const;
#ifdef __CUDACC__
                __device__ __forceinline MLPL<_T>* GetLayers() const;
#endif
                __host__ __device__ __forceinline void SetLayers(MLPL<_T>* Layers);
                __host__ __device__ __forceinline MLPL<_T> GetLayer(size_t LayerIndex) const;
                __host__ __device__ __forceinline void SetLayer(size_t LayerIndex, MLPL<_T> Layer);

                __host__ __device__ __forceinline size_t InputLength();
                __host__ __device__ __forceinline size_t OutputLength();

                __host__ __forceinline _T* Run(_T* Input) const;

                __host__ __device__ __forceinline MLP<_T> Clone() const;
                __host__ __device__ __forceinline void Randomize(_T Scalar, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline MLP<_T> Reproduce(_T Scalar, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ __forceinline MLP<_T> Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) const;
            private:
                size_t len;
                MLPL<_T>* lyrs;
                activationFunction_t<_T> actnFunc;
            };
        }
    }
}

template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLP<_T>::MLP(size_t Length, activationFunction_t<_T> ActivationFunction) {
    len = Length;
    actnFunc = ActivationFunction;
#ifdef __CUDA_ARCH__
    lyrs = new MLPL<_T>[Length];
#else
    ThrowIfBad(cudaMalloc(&lyrs, Length * sizeof(MLPL<_T>)));
#endif
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLP<_T>::MLP(size_t Length, activationFunction_t<_T> ActivationFunction, MLPL<_T>* Layers)
    : MLP(Length, ActivationFunction) {
    SetLayers(Layers);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::Dispose() {
    for (size_t i = 0; i < len; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->Dispose();
#else
        GetLayer(i).Dispose();
#endif
    }
#ifdef __CUDA_ARCH__
    delete[] lyrs;
#else
    ThrowIfBad(cudaFree(lyrs));
#endif
}
template <typename _T>
__host__ __forceinline _T* BrendanCUDA::AI::MLP::MLP<_T>::Run(_T* Input) const {
    if (!len) return 0;
#ifdef __CUDA_ARCH__
    MLPL<_T>* l = Layer(0);
    Input = l->Run(Input);
    details::RunActivationFunctionOnArray(Span<_T>(Input, l->OutputLength()), actnFunc);
#else
    MLPL<_T> l = GetLayer(0);
    Input = l.Run(Input);
    details::RunActivationFunctionOnArray(Span<_T>(Input, l.OutputLength()), actnFunc);
#endif
    for (size_t i = 1; i < len; ++i) {
#ifdef __CUDA_ARCH__
        l = Layer(i);
        _T* nxt = l->Run(Input);
        details::RunActivationFunctionOnArray(Span<_T>(nxt, l->OutputLength()), actnFunc);
        delete[] Input;
#else
        l = GetLayer(i);
        _T* nxt = l.Run(Input);
        details::RunActivationFunctionOnArray(Span<_T>(nxt, l.OutputLength()), actnFunc);
        ThrowIfBad(cudaFree(Input));
#endif
        Input = nxt;
    }
    return Input;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::Layers() const {
    return lyrs;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::Layer(size_t LayerIndex) const {
    return &lyrs[LayerIndex];
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::AI::MLP::MLP<_T>::LayerCount() const {
    return len;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::activationFunction_t<_T> BrendanCUDA::AI::MLP::MLP<_T>::ActivationFunction() const {
    return actnFunc;
}
template <typename _T>
template <bool _CopyToHost>
__host__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::GetLayers() const {
    MLPL<_T>* p;
    if constexpr (_CopyToHost) {
        p = new MLPL<_T>[len];
        ThrowIfBad(cudaMemcpy(p, lyrs, sizeof(MLPL<_T>) * len, cudaMemcpyDeviceToHost));
    }
    else {
        ThrowIfBad(cudaMalloc(&p, sizeof(MLPL<_T>) * len));
        ThrowIfBad(cudaMemcpy(p, lyrs, sizeof(MLPL<_T>) * len, cudaMemcpyDeviceToDevice));
    }
    return p;
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::GetLayers() const {
    MLPL<_T>* p = new MLPL<_T>[len];
    deviceMemcpy(p, lyrs, sizeof(MLPL<_T>) * len);
    return p;
}
#endif
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::SetLayers(MLPL<_T>* Layers) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(lyrs, Layers, sizeof(MLPL<_T>) * len);
#else
    ThrowIfBad(cudaMemcpy(lyrs, Layers, sizeof(MLPL<_T>) * len, cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLP<_T>::GetLayer(size_t LayerIndex) const {
#ifdef __CUDA_ARCH__
    return lyrs[LayerIndex];
#else
    MLPL<_T> r;
    ThrowIfBad(cudaMemcpy(&r, &lyrs[LayerIndex], sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return r;
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::SetLayer(size_t LayerIndex, MLPL<_T> Layer) {
#ifdef __CUDA_ARCH__
    lyrs[LayerIndex] = Layer;
#else
    ThrowIfBad(cudaMemcpy(&lyrs[LayerIndex], &Layer, sizeof(MLPL<_T>), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::AI::MLP::MLP<_T>::InputLength() {
    if (len == 0) {
        return 0;
    }
#ifdef __CUDA_ARCH__
    return Layer(0)->InputLength();
#else
    return GetLayer(0).InputLength();
#endif
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::AI::MLP::MLP<_T>::OutputLength() {
    if (len == 0) {
        return 0;
    }
#ifdef __CUDA_ARCH__
    return Layer(len - 1)->OutputLength();
#else
    return GetLayer(len - 1).OutputLength();
#endif
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLP<_T> BrendanCUDA::AI::MLP::MLP<_T>::Clone() const {
    MLP<_T> m(len, actnFunc);
    for (size_t i = 0; i < len; ++i) {
#ifdef __CUDA_ARCH__
        m.SetLayer(i, Layer(i)->Clone());
#else
        m.SetLayer(i, GetLayer(i).Clone());
#endif
    }
    return m;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::Randomize(_T Scalar, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < len; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->Randomize(Scalar, RNG);
#else
        GetLayer(i).Randomize(Scalar, RNG);
#endif
    }
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < len; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->Randomize(Scalar, LowerBound, UpperBound, RNG);
#else
        GetLayer(i).Randomize(Scalar, LowerBound, UpperBound, RNG);
#endif
    }
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLP<_T> BrendanCUDA::AI::MLP::MLP<_T>::Reproduce(_T Scalar, Random::AnyRNG<uint64_t> RNG) const {
    MLP<_T> n = Clone();
    n.Randomize(Scalar, RNG);
    return n;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::AI::MLP::MLP<_T> BrendanCUDA::AI::MLP::MLP<_T>::Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) const {
    MLP<_T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, RNG);
    return n;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::ZeroOverwrite() {
    for (size_t i = 0; i < len; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->ZeroOverwrite();
#else
        GetLayer(i).ZeroOverwrite();
#endif
    }
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::RandomOverwrite(Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < len; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->RandomOverwrite(RNG);
#else
        GetLayer(i).RandomOverwrite(RNG);
#endif
    }
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLP::MLP<_T>::RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < len; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->RandomOverwrite(LowerBound, UpperBound, RNG);
#else
        GetLayer(i).RandomOverwrite(LowerBound, UpperBound, RNG);
#endif
    }
}