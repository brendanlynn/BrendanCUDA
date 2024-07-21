#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include "brendancuda_rand_anyrng.h"
#include <random>
#include "brendancuda_ai.h"
#include "brendancuda_mathfuncs.h"
#include "BSerializer/Serializer.h"

namespace BrendanCUDA {
    namespace AI {
        namespace MLP {
            template <typename _T, activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
            struct FixedMLPL;
        }
    }
    namespace details {
        template <size_t _Index, typename _T, AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _ContinuedOutputCounts>
        struct MLPLayerType;
        template <size_t _Index, typename _T, AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
        struct MLPLayerType<_Index, _T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> {
            using type = typename MLPLayerType<_Index - 1, _T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::type;
        };
        template <typename _T, AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
        struct MLPLayerType<0, _T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> {
            using type = AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count>;
        };
        template <size_t _Index, typename _T, AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
        struct MLPLayerType<_Index, _T, _ActivationFunction, _InputCount, _Output1Count> {
            static_assert(!_Index, "_Index is out of bounds.");
            using type = AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count>;
        };

        template <size_t _Index, typename _T, AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _ContinuedOutputCounts>
        using mlpLayerType_t = MLPLayerType<_Index, _T, _ActivationFunction, _InputCount, _Output1Count, _ContinuedOutputCounts...>;
    }
    namespace AI {
        namespace MLP {
            template <typename _T, activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
            struct FixedMLPL final {
                static_assert(std::is_same<_T, float>::value || std::is_same<_T, double>::value, "_T must be either float or double.");
                static_assert(_InputCount, "_InputCount must be greater than 0.");
                static_assert(_OutputCount, "_OutputCount must be greater than 0.");

                _T weights[_InputCount][_OutputCount];
                _T bias[_OutputCount];

                __host__ __device__ void FillWith0();
                __host__ __device__ void FillWithRandom(Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void ChangeWithRandom(_T Scalar, Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void Run(const _T* Input, _T* Output) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount> Deserialize(const void*& Data);
            };
            template <typename _T, activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
            struct FixedMLP;
            template <typename _T, activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
            struct FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> final {
                static_assert(std::is_same<_T, float>::value || std::is_same<_T, double>::value, "_T must be either float or double.");

                template <size_t _Index>
                using layerType_t = details::mlpLayerType_t<_Index, _T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>;
                
                FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count> layer;
                FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...> nextLayers;

                __host__ __device__ void FillWith0();
                __host__ __device__ void FillWithRandom(Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void ChangeWithRandom(_T Scalar, Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void Run(const _T* Input, _T* Intermediate1, _T* Intermediate2, _T* Output) const;
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer();

                static constexpr size_t InputCount();
                static constexpr size_t OutputCount();
                static constexpr size_t Intermediate0Count();
                static constexpr size_t Intermediate1Count();
                static constexpr size_t MaxLayerOutputCount();

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>& Value);
            };
            template <typename _T, activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
            struct FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count> final {
                static_assert(std::is_same<_T, float>::value || std::is_same<_T, double>::value, "_T must be either float or double.");

                template <size_t _Index>
                using layerType_t = details::mlpLayerType_t<_Index, _T, _ActivationFunction, _InputCount, _Output1Count>;

                FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count> layer;

                __host__ __device__ void FillWith0();
                __host__ __device__ void FillWithRandom(Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void ChangeWithRandom(_T Scalar, Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ void Run(const _T* Input, _T* Intermediate1, _T* Intermediate2, _T* Output) const;
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer();

                static constexpr size_t InputCount();
                static constexpr size_t OutputCount();
                static constexpr size_t Intermediate0Count();
                static constexpr size_t Intermediate1Count();
                static constexpr size_t MaxLayerOutputCount();

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count> Deserialize(const void*& Data);
            };
        }
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::FillWith0() {
    for (size_t i = 0; i < _OutputCount; ++i) {
        for (size_t j = 0; j < _InputCount; ++j) {
            weights[i][j] = 0.;
        }
        bias[i] = 0.;
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::FillWithRandom(Random::AnyRNG<uint32_t> RNG) {
    std::uniform_real_distribution<_T> dis(-1., 1.);

    for (size_t i = 0; i < _OutputCount; ++i) {
        for (size_t j = 0; j < _InputCount; ++j) {
            weights[i][j] = dis(RNG);
        }
        bias[i] = dis(RNG);
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::ChangeWithRandom(_T Scalar, Random::AnyRNG<uint32_t> RNG) {
    std::uniform_real_distribution<_T> dis(-Scalar, Scalar);

    for (size_t i = 0; i < _OutputCount; ++i) {
        for (size_t j = 0; j < _InputCount; ++j) {
            weights[i][j] += dis(RNG);
        }
        bias[i] += dis(RNG);
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint32_t> RNG) {
    std::uniform_real_distribution<_T> dis(-Scalar, Scalar);

    for (size_t i = 0; i < _OutputCount; ++i) {
        for (size_t j = 0; j < _InputCount; ++j) {
            _T& v = weights[i][j];
            v = Math::clamp(v + dis(RNG), LowerBound, UpperBound);
        }
        _T& v = bias[i];
        v = Math::clamp(v + dis(RNG), LowerBound, UpperBound);
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::Run(const _T* Input, _T* Output) const {
    for (size_t j = 0; j < _OutputCount; ++j) {
        float v = bias[j];
        for (size_t i = 0; i < _InputCount; ++i) {
            v += weights[i][j] * Input[i];
        }
        Output[j] = _ActivationFunction(v);
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
size_t BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::SerializedSize() const {
    return sizeof(FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
void BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::Serialize(void*& Data) const {
    constexpr size_t s = sizeof(FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>);
    memcpy(Data, this, s);
    Data = ((uint8_t*)Data) + s;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
static BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount> BrendanCUDA::AI::MLP::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>::Deserialize(const void*& Data) {
    constexpr size_t s = sizeof(FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>);
    FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount> mlp;
    memcpy(&mlp, Data, s);
    Data = ((uint8_t*)Data) + s;
    return mlp;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::FillWith0() {
    layer.FillWith0();
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::FillWithRandom(Random::AnyRNG<uint32_t> RNG) {
    layer.FillWithRandom(RNG);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::ChangeWithRandom(_T Scalar, Random::AnyRNG<uint32_t> RNG) {
    layer.ChangeWithRandom(Scalar, RNG);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint32_t> RNG) {
    layer.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::FillWith0() {
    layer.FillWith0();
    nextLayers.FillWith0();
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::FillWithRandom(Random::AnyRNG<uint32_t> RNG) {
    layer.FillWithRandom(RNG);
    nextLayers.FillWithRandom(RNG);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::ChangeWithRandom(_T Scalar, Random::AnyRNG<uint32_t> RNG) {
    layer.ChangeWithRandom(Scalar, RNG);
    nextLayers.ChangeWithRandom(Scalar, RNG);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint32_t> RNG) {
    layer.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
    nextLayers.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Run(const _T* Input, _T* Intermediate1, _T* Intermediate2, _T* Output) const {
    _T* i1 = Intermediate1 ? Intermediate1 : new _T[Intermediate0Count()];
    _T* i2 = Intermediate2 ? Intermediate2 : new _T[Intermediate1Count()];
    
    layer.Run(Input, i1);
    nextLayers.Run(i1, i2, i1, Output);

    if (!Intermediate1) delete[] i1;
    if (!Intermediate2) delete[] i2;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
template <size_t _Index>
__host__ __device__ BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::layerType_t<_Index>& BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Layer() {
    if constexpr (_Index) {
        return nextLayers.Layer<_Index - 1>();
    }
    else {
        return layer;
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
__host__ __device__ void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::Run(const _T* Input, _T* Intermediate1, _T* Intermediate2, _T* Output) const {
    layer.Run(Input, Output);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
template <size_t _Index>
__host__ __device__ BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::layerType_t<_Index>& BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::Layer() {
    static_assert(!_Index, "_Index is out of bounds.");
    return layer;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::InputCount() {
    return _InputCount;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::InputCount() {
    return _InputCount;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::OutputCount() {
    return BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::OutputCount();
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::OutputCount() {
    return _Output1Count;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Intermediate0Count() {
    return std::max(_Output1Count, BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Intermediate1Count());
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::Intermediate0Count() {
    return 0;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Intermediate1Count() {
    return BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Intermediate0Count();
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::Intermediate1Count() {
    return 0;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::MaxLayerOutputCount() {
    constexpr size_t maxNextLayers = FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::MaxLayerOutputCount();
    return _Output1Count > maxNextLayers ? _Output1Count : maxNextLayers;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::SerializedSize() const {
    return sizeof(FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Serialize(void*& Data) const {
    BSerializer::SerializeRaw(Data, *this);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Deserialize(const void*& Data) {
    return BSerializer::DeserializeRaw<FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>>(Data);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Deserialize(const void*& Data, FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>& Value) {
    BSerializer::DeserializeRaw<FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>>(Data, Value);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
constexpr size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::MaxLayerOutputCount() {
    return _Output1Count;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
size_t BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::SerializedSize() const {
    return sizeof(FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>);
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
void BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::Serialize(void*& Data) const {
    constexpr size_t s = sizeof(FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>);
    memcpy(Data, this, s);
    Data = ((uint8_t*)Data) + s;
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count>
static BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count> BrendanCUDA::AI::MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>::Deserialize(const void*& Data) {
    constexpr size_t s = sizeof(FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>);
    FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count> mlp;
    memcpy(&mlp, Data, s);
    Data = ((uint8_t*)Data) + s;
    return mlp;
}