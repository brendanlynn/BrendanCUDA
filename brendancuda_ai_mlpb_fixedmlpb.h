#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include "brendancuda_rand_anyrng.h"
#include <random>
#include "brendancuda_ai.h"
#include "brendancuda_mathfuncs.h"
#include "brendancuda_rand_randomizer.h"
#include "BSerializer/Serializer.h"

namespace BrendanCUDA {
    namespace AI {
        namespace MLPB {
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
            struct FixedMLPBL;
        }
    }
    namespace details {
        template <size_t _Index, std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral... _TsContinuedOutputs>
        struct MLPBLayerType;
        template <size_t _Index, std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
        struct MLPBLayerType<_Index, _TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...> {
            using type = typename MLPBLayerType<_Index - 1, _TOutput1, _TOutput2, _TsContinuedOutputs...>::type;
        };
        template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
        struct MLPBLayerType<0, _TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...> {
            using type = AI::MLPB::FixedMLPBL<_TInput, _TOutput1>;
        };
        template <size_t _Index, std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
        struct MLPBLayerType<_Index, _TInput, _TOutput1> {
            static_assert(!_Index, "_Index is out of bounds.");
            using type = AI::MLPB::FixedMLPBL<_TInput, _TOutput1>;
        };

        template <size_t _Index, std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral... _TsContinuedOutputs>
        using mlpbLayerType_t = MLPBLayerType<_Index, _TInput, _TOutput1, _TsContinuedOutputs...>;

        template <std::integral _T>
        constexpr size_t BitCount = sizeof(_T) << 3;
    }
    namespace AI {
        namespace MLPB {
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
            struct FixedMLPBL final {
                _TInput weights[details::BitCount<_TOutput>];
                _TOutput bias;

                __host__ __device__ void FillWith0();
                __host__ __device__ void FillWithRandom(Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ _TOutput Run(_TInput Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static FixedMLPBL<_TInput, _TOutput> Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, FixedMLPBL<_TInput, _TOutput>& Value);
            };
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral... _LayerCounts>
            struct FixedMLPB;
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
            struct FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...> final {
                template <size_t _Index>
                using layerType_t = details::mlpbLayerType_t<_Index, _TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>;

                using input_t = _TInput;
                using output_t = FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...>::output_t;
                
                FixedMLPBL<_TInput, _TOutput1> layer;
                FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...> nextLayers;

                __host__ __device__ void FillWith0();
                __host__ __device__ void FillWithRandom(Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ output_t Run(_TInput Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer();

                static constexpr size_t LayerCount();

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...> Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>& Value);
            };
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
            struct FixedMLPB<_TInput, _TOutput1> final {
                template <size_t _Index>
                using layerType_t = details::mlpbLayerType_t<_Index, _TInput, _TOutput1>;

                using input_t = _TInput;
                using output_t = _TOutput1;

                FixedMLPBL<_TInput, _TOutput1> layer;

                __host__ __device__ void FillWith0();
                __host__ __device__ void FillWithRandom(Random::AnyRNG<uint32_t> RNG);
                __host__ __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ output_t Run(_TInput Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer();

                static constexpr size_t LayerCount();

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static FixedMLPB<_TInput, _TOutput1> Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, FixedMLPB<_TInput, _TOutput1>& Value);
            };
        }
    }
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ void BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::FillWith0() {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i) {
        weights[i] = (_TInput)0;
    }
    bias = (_TInput)0;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ void BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::FillWithRandom(Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i) {
        weights[i] = (_TInput)RNG();
    }
    bias = (_TInput)RNG();
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = BrendanCUDA::Random::RandomizeWFlips(weights[i], WeightsFlipProb, RNG);
    bias = BrendanCUDA::Random::RandomizeWFlips(bias, BiasFlipProb, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = BrendanCUDA::Random::RandomizeWTargets(weights[i], WeightsFlipProb, RNG);
    bias = BrendanCUDA::Random::RandomizeWTargets(bias, BiasFlipProb, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = BrendanCUDA::Random::RandomizeWMutations(weights[i], WeightsProbOf1, RNG);
    bias = BrendanCUDA::Random::RandomizeWMutations(bias, BiasProbOf1, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ _TOutput BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Run(_TInput Input) const {
    _TOutput o = bias;
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i) {
        if (Input & weights[i]) o |= ((_TOutput)1) << i;
    }
    return o;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RunG(uint64_t Input) const {
    return (uint64_t)Run((_TInput)Input);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
size_t BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::SerializedSize() const {
    return sizeof(FixedMLPBL<_TInput, _TOutput>);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
void BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Serialize(void*& Data) const {
    BSerializer::SerializeRaw(Data, *this);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Deserialize(const void*& Data) {
    return BSerializer::DeserializeRaw<FixedMLPBL<_TInput, _TOutput>>(Data);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
void BrendanCUDA::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Deserialize(const void*& Data, FixedMLPBL<_TInput, _TOutput>& Value) {
    BSerializer::DeserializeRaw(Data, Value);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::FillWith0() {
    layer.FillWith0();
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::FillWithRandom(Random::AnyRNG<uint32_t> RNG) {
    layer.FillWithRandom(RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) {
    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ auto BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Run(_TInput Input) const -> output_t {
    return (output_t)RunG((uint64_t)Input);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RunG(uint64_t Input) const {
    Input = layer.RunG(Input);
    return Input;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::FillWith0() {
    layer.FillWith0();
    nextLayers.FillWith0();
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::FillWithRandom(Random::AnyRNG<uint32_t> RNG) {
    layer.FillWithRandom(RNG);
    nextLayers.FillWithRandom(RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
    nextLayers.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
    nextLayers.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) {
    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
    nextLayers.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ auto BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Run(_TInput Input) const -> output_t {
    return (output_t)RunG((uint64_t)Input);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RunG(uint64_t Input) const {
    Input = layer.RunG(Input);
    Input = nextLayers.RunG(Input);
    return Input;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <size_t _Index>
__host__ __device__ BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::layerType_t<_Index>& BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Layer() {
    if constexpr (_Index) {
        return nextLayers.Layer<_Index - 1>();
    }
    else {
        return layer;
    }
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <size_t _Index>
__host__ __device__ BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::layerType_t<_Index>& BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Layer() {
    static_assert(!_Index, "_Index is out of bounds.");
    return layer;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
constexpr size_t BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::LayerCount() {
    return FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...>::LayerCount() + 1;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
size_t BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::SerializedSize() const {
    return sizeof(FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Serialize(void*& Data) const {
    BSerializer::SerializeRaw(Data, *this);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...> BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Deserialize(const void*& Data) {
    return BSerializer::DeserializeRaw<FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>>(Data);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Deserialize(const void*& Data, FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>& Value) {
    BSerializer::DeserializeRaw(Data, Value);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
constexpr size_t BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::LayerCount() {
    return 1;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
size_t BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::SerializedSize() const {
    return sizeof(FixedMLPB<_TInput, _TOutput1>);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Serialize(void*& Data) const {
    BSerializer::SerializeRaw(*this, Data);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1> BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Deserialize(const void*& Data) {
    return BSerializer::DeserializeRaw<FixedMLPB<_TInput, _TOutput1>>(Data);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
void BrendanCUDA::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Deserialize(const void*& Data, FixedMLPB<_TInput, _TOutput1>& Value) {
    BSerializer::DeserializeRaw(Data, Value);
}