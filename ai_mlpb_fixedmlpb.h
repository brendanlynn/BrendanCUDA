#pragma once

#include "ai.h"
#include "BSerializer/Serializer.h"
#include "curandkernelgens.h"
#include "details_getintbin.h"
#include "mathfuncs.h"
#include "rand_anyrng.h"
#include "rand_randomizer.h"
#include <cuda_runtime.h>
#include <random>
#include <type_traits>

namespace bcuda {
    namespace ai {
        namespace mlpb {
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
            using type = ai::mlpb::FixedMLPBL<_TInput, _TOutput1>;
        };
        template <size_t _Index, std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
        struct MLPBLayerType<_Index, _TInput, _TOutput1> {
            static_assert(!_Index, "_Index is out of bounds.");
            using type = ai::mlpb::FixedMLPBL<_TInput, _TOutput1>;
        };

        template <size_t _Index, std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral... _TsContinuedOutputs>
        using mlpbLayerType_t = MLPBLayerType<_Index, _TInput, _TOutput1, _TsContinuedOutputs...>;

        template <std::integral _T>
        constexpr size_t bitCount = sizeof(_T) << 3;
    }
    namespace ai {
        namespace mlpb {
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
            struct FixedMLPBL {
            private:
                using this_t = FixedMLPBL<_TInput, _TOutput>;
            public:
                _TInput weights[details::bitCount<_TOutput>];
                _TOutput bias;

                __host__ __device__ void FillWith0() {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = (_TInput)0;
                    bias = (_TOutput)0;
                }
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void FillWithRandom(_TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = details::GetIntBin<_TInput>(RNG);
                    bias = details::GetIntBin<_TOutput>(RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ void FillWithRandom(_TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = details::GetIntBin<_TInput>(RNG);
                    bias = details::GetIntBin<_TOutput>(RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = bcuda::rand::RandomizeWFlips(weights[i], WeightsFlipProb, RNG);
                    bias = bcuda::rand::RandomizeWFlips(bias, BiasFlipProb, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = bcuda::rand::RandomizeWFlips(weights[i], WeightsFlipProb, RNG);
                    bias = bcuda::rand::RandomizeWFlips(bias, BiasFlipProb, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = bcuda::rand::RandomizeWTargets(weights[i], WeightsFlipProb, RNG);
                    bias = bcuda::rand::RandomizeWTargets(bias, BiasFlipProb, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = bcuda::rand::RandomizeWTargets(weights[i], WeightsFlipProb, RNG);
                    bias = bcuda::rand::RandomizeWTargets(bias, BiasFlipProb, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = bcuda::rand::RandomizeWMutations(weights[i], WeightsProbOf1, RNG);
                    bias = bcuda::rand::RandomizeWMutations(bias, BiasProbOf1, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        weights[i] = bcuda::rand::RandomizeWMutations(weights[i], WeightsProbOf1, RNG);
                    bias = bcuda::rand::RandomizeWMutations(bias, BiasProbOf1, RNG);
                }
#endif
                __host__ __device__ _TOutput Run(_TInput Input) const {
                    _TOutput o = bias;
                    for (size_t i = 0; i < details::bitCount<_TOutput>; ++i)
                        if (Input & weights[i]) o |= ((_TOutput)1) << i;
                    return o;
                }
                __host__ __device__ uint64_t RunG(uint64_t Input) const {
                    return (uint64_t)Run((_TInput)Input);
                }

                size_t SerializedSize() const {
                    return sizeof(this_t);
                }
                void Serialize(void*& Data) const {
                    BSerializer::SerializeArray(Data, weights, details::bitCount<_TOutput>);
                    BSerializer::Serialize(Data, bias);
                }
                static this_t Deserialize(const void*& Data) {
                    uint8_t bytes[sizeof(this_t)];
                    Deserialize(Data, &bytes);
                    return *(this_t*)&bytes;
                }
                static void Deserialize(const void*& Data, void* ObjMem) {
                    this_t& obj = *(this_t*)ObjMem;
                    BSerializer::DeserializeArray(Data, obj.weights, details::bitCount<_TOutput>);
                    BSerializer::Deserialize(Data, &obj.bias);
                }
            };
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral... _LayerCounts>
            struct FixedMLPB;
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
            struct FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...> {
            private:
                using this_t = FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>;
            public:
                template <size_t _Index>
                using layerType_t = details::mlpbLayerType_t<_Index, _TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>;

                using input_t = _TInput;
                using output_t = typename FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...>::output_t;
                
                FixedMLPBL<_TInput, _TOutput1> layer;
                FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...> nextLayers;

                __host__ __device__ void FillWith0() {
                    layer.FillWith0();
                    nextLayers.FillWith0();
                }
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                    nextLayers.FillWithRandom(RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                    nextLayers.FillWithRandom(RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
                    nextLayers.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
                    nextLayers.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
                    nextLayers.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
                    nextLayers.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
                    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
                    nextLayers.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#endif
                __host__ __device__ output_t Run(_TInput Input) const {
                    return (output_t)RunG((uint64_t)Input);
                }
                __host__ __device__ uint64_t RunG(uint64_t Input) const {
                    Input = layer.RunG(Input);
                    Input = nextLayers.RunG(Input);
                    return Input;
                }
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer() {
                    if constexpr (_Index) {
                        return nextLayers.Layer<_Index - 1>();
                    }
                    else {
                        return layer;
                    }
                }

                static constexpr size_t LayerCount() {
                    return sizeof...(_TsContinuedOutputs) + 2;
                }

                size_t SerializedSize() const {
                    return sizeof(this_t);
                }
                void Serialize(void*& Data) const {
                    layer.Serialize(Data);
                    nextLayers.Serialize(Data);
                }
                static this_t Deserialize(const void*& Data) {
                    uint8_t bytes[sizeof(this_t)];
                    Deserialize(Data, &bytes);
                    return *(this_t*)&bytes;
                }
                static void Deserialize(const void*& Data, void* ObjMem) {
                    this_t& obj = &(this_t*)ObjMem;
                    BSerializer::Deserialize(Data, &obj.layer);
                    BSerializer::Deserialize(Data, &obj.nextLayers);
                }
            };
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
            struct FixedMLPB<_TInput, _TOutput1> {
            private:
                using this_t = FixedMLPB<_TInput, _TOutput1>;
            public:
                template <size_t _Index>
                using layerType_t = details::mlpbLayerType_t<_Index, _TInput, _TOutput1>;

                using input_t = _TInput;
                using output_t = _TOutput1;

                FixedMLPBL<_TInput, _TOutput1> layer;

                __host__ __device__ void FillWith0() {
                    layer.FillWith0();
                }
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
                    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
                    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
                    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
                }
#endif
                __host__ __device__ output_t Run(_TInput Input) const {
                    return (output_t)RunG((uint64_t)Input);
                }
                __host__ __device__ uint64_t RunG(uint64_t Input) const {
                    Input = layer.RunG(Input);
                    return Input;
                }
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer() {
                    static_assert(!_Index, "_Index is out of bounds.");
                    return layer;
                }

                static constexpr size_t LayerCount() {
                    return 1;
                }

                size_t SerializedSize() const {
                    return sizeof(this_t);
                }
                void Serialize(void*& Data) const {
                    layer.Serialize(Data);
                }
                static this_t Deserialize(const void*& Data) {
                    uint8_t bytes[sizeof(this_t)];
                    Deserialize(Data, &bytes);
                    return *(this_t*)&bytes;
                }
                static void Deserialize(const void*& Data, void* ObjMem) {
                    this_t& obj = &(this_t*)ObjMem;
                    BSerializer::Deserialize(Data, &obj.layer);
                }
            };
        }
    }
}