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

namespace brendancuda {
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
            struct FixedMLPBL {
            private:
                using this_t = FixedMLPBL<_TInput, _TOutput>;
            public:
                _TInput weights[details::BitCount<_TOutput>];
                _TOutput bias;

                __host__ __device__ void FillWith0();
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void FillWithRandom(_TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ void FillWithRandom(_TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#endif
                __host__ __device__ _TOutput Run(_TInput Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static this_t Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, void* ObjMem);
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
                using output_t = FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...>::output_t;
                
                FixedMLPBL<_TInput, _TOutput1> layer;
                FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...> nextLayers;

                __host__ __device__ void FillWith0();
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void FillWithRandom(_TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ void FillWithRandom(_TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#endif
                __host__ __device__ output_t Run(_TInput Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer();

                static constexpr size_t LayerCount();

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static this_t Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, void* ObjMem);
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

                __host__ __device__ void FillWith0();
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void FillWithRandom(_TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ void FillWithRandom(_TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#endif
                __host__ __device__ output_t Run(_TInput Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                template <size_t _Index>
                __host__ __device__ layerType_t<_Index>& Layer();

                static constexpr size_t LayerCount();

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static this_t Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, void* ObjMem);
            };
        }
    }
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::FillWith0() {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i) {
        weights[i] = (_TInput)0;
    }
    bias = (_TOutput)0;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <std::uniform_random_bit_generator _TRNG>
__host__ void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::FillWithRandom(_TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i) {
        weights[i] = details::GetIntBin<_TInput>(RNG);
    }
    bias = details::GetIntBin<_TOutput>(RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <brendancuda::KernelCurandState _TRNG>
__device__ void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::FillWithRandom(_TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i) {
        weights[i] = details::GetIntBin<_TInput>(RNG);
    }
    bias = details::GetIntBin<_TOutput>(RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = brendancuda::Random::RandomizeWFlips(weights[i], WeightsFlipProb, RNG);
    bias = brendancuda::Random::RandomizeWFlips(bias, BiasFlipProb, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = brendancuda::Random::RandomizeWFlips(weights[i], WeightsFlipProb, RNG);
    bias = brendancuda::Random::RandomizeWFlips(bias, BiasFlipProb, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = brendancuda::Random::RandomizeWTargets(weights[i], WeightsFlipProb, RNG);
    bias = brendancuda::Random::RandomizeWTargets(bias, BiasFlipProb, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = brendancuda::Random::RandomizeWTargets(weights[i], WeightsFlipProb, RNG);
    bias = brendancuda::Random::RandomizeWTargets(bias, BiasFlipProb, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = brendancuda::Random::RandomizeWMutations(weights[i], WeightsProbOf1, RNG);
    bias = brendancuda::Random::RandomizeWMutations(bias, BiasProbOf1, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i)
        weights[i] = brendancuda::Random::RandomizeWMutations(weights[i], WeightsProbOf1, RNG);
    bias = brendancuda::Random::RandomizeWMutations(bias, BiasProbOf1, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ _TOutput brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Run(_TInput Input) const {
    _TOutput o = bias;
    for (size_t i = 0; i < details::BitCount<_TOutput>; ++i) {
        if (Input & weights[i]) o |= ((_TOutput)1) << i;
    }
    return o;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ uint64_t brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::RunG(uint64_t Input) const {
    return (uint64_t)Run((_TInput)Input);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
size_t brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::SerializedSize() const {
    return sizeof(this_t);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Serialize(void*& Data) const {
    BSerializer::SerializeArray(Data, weights, details::BitCount<_TOutput>);
    BSerializer::Serialize(Data, bias);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput> brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Deserialize(const void*& Data) {
    uint8_t bytes[sizeof(this_t)];
    Deserialize(Data, &bytes);
    return *(this_t*)&bytes;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
void brendancuda::AI::MLPB::FixedMLPBL<_TInput, _TOutput>::Deserialize(const void*& Data, void* ObjMem) {
    this_t& obj = *(this_t*)ObjMem;
    BSerializer::DeserializeArray(Data, obj.weights, details::BitCount<_TOutput>);
    BSerializer::Deserialize(Data, &obj.bias);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::FillWith0() {
    layer.FillWith0();
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <std::uniform_random_bit_generator _TRNG>
__host__ void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::FillWithRandom(_TRNG& RNG) {
    layer.FillWithRandom(RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <brendancuda::KernelCurandState _TRNG>
__device__ void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::FillWithRandom(_TRNG& RNG) {
    layer.FillWithRandom(RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ auto brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Run(_TInput Input) const -> output_t {
    return (output_t)RunG((uint64_t)Input);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
__host__ __device__ uint64_t brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::RunG(uint64_t Input) const {
    Input = layer.RunG(Input);
    return Input;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::FillWith0() {
    layer.FillWith0();
    nextLayers.FillWith0();
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <std::uniform_random_bit_generator _TRNG>
__host__ void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::FillWithRandom(_TRNG& RNG) {
    layer.FillWithRandom(RNG);
    nextLayers.FillWithRandom(RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <brendancuda::KernelCurandState _TRNG>
__device__ void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::FillWithRandom(_TRNG& RNG) {
    layer.FillWithRandom(RNG);
    nextLayers.FillWithRandom(RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
    nextLayers.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
    nextLayers.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
    nextLayers.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWTargets(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    layer.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
    nextLayers.RandomizeWTargets(WeightsFlipProb, BiasFlipProb, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <std::uniform_random_bit_generator _TRNG>
__host__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
    nextLayers.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
}

#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <brendancuda::KernelCurandState _TRNG>
__device__ __forceinline void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    layer.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
    nextLayers.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
}
#endif

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ auto brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Run(_TInput Input) const -> output_t {
    return (output_t)RunG((uint64_t)Input);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
__host__ __device__ uint64_t brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::RunG(uint64_t Input) const {
    Input = layer.RunG(Input);
    Input = nextLayers.RunG(Input);
    return Input;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
template <size_t _Index>
__host__ __device__ brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::layerType_t<_Index>& brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Layer() {
    if constexpr (_Index) {
        return nextLayers.Layer<_Index - 1>();
    }
    else {
        return layer;
    }
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
template <size_t _Index>
__host__ __device__ brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::layerType_t<_Index>& brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Layer() {
    static_assert(!_Index, "_Index is out of bounds.");
    return layer;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
constexpr size_t brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::LayerCount() {
    return FixedMLPB<_TOutput1, _TOutput2, _TsContinuedOutputs...>::LayerCount() + 1;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
size_t brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::SerializedSize() const {
    return sizeof(this_t);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Serialize(void*& Data) const {
    layer.Serialize(Data);
    nextLayers.Serialize(Data);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...> brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Deserialize(const void*& Data) {
    uint8_t bytes[sizeof(this_t)];
    Deserialize(Data, &bytes);
    return *(this_t*)&bytes;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1, std::unsigned_integral _TOutput2, std::unsigned_integral... _TsContinuedOutputs>
void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1, _TOutput2, _TsContinuedOutputs...>::Deserialize(const void*& Data, void* ObjMem) {
    this_t& obj = &(this_t*)ObjMem;
    BSerializer::Deserialize(Data, &obj.layer);
    BSerializer::Deserialize(Data, &obj.nextLayers);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
constexpr size_t brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::LayerCount() {
    return 1;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
size_t brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::SerializedSize() const {
    return sizeof(this_t);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Serialize(void*& Data) const {
    layer.Serialize(Data);
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1> brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Deserialize(const void*& Data) {
    uint8_t bytes[sizeof(this_t)];
    Deserialize(Data, &bytes);
    return *(this_t*)&bytes;
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput1>
void brendancuda::AI::MLPB::FixedMLPB<_TInput, _TOutput1>::Deserialize(const void*& Data, void* ObjMem) {
    this_t& obj = &(this_t*)ObjMem;
    BSerializer::Deserialize(Data, &obj.layer);
}