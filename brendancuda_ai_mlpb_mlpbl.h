#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <bit>
#include <concepts>
#include "BSerializer/Serializer.h"
#include "brendancuda_rand_anyrng.h"
#include "brendancuda_ai.h"
#include "brendancuda_binary_basic.h"
#include "brendancuda_rand_bits.h"
#include "brendancuda_errorhelp.h"
#include "brendancuda_dcopy.cuh"
#include "brendancuda_rand_sseed.h"

namespace BrendanCUDA {
    namespace details {
        __host__ __device__ void applyTargetFlipsOnArray(uint64_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
        __host__ __device__ void applyTargetFlipsOnArray(uint32_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
        __host__ __device__ void applyTargetFlipsOnArray(uint16_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
        __host__ __device__ void applyTargetFlipsOnArray(uint8_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
    }
    namespace AI {
        namespace MLPB {
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
            class MLPBL final {
            public:
                __host__ __device__ __forceinline MLPBL();
                __host__ __device__ __forceinline MLPBL(_TInput* Weights, _TOutput* Bias);

                __host__ __device__ static __forceinline MLPBL<_TInput, _TOutput> CloneFrom(_TInput* Weights, _TOutput* Bias);

                __host__ __device__ __forceinline void Dispose();

                __host__ __device__ __forceinline _TInput* Weights() const;
                __host__ __device__ __forceinline _TInput* Weight(size_t Index) const;

                __host__ __device__ __forceinline _TOutput* Bias() const;

                template <bool _CopyToHost>
                __host__ __forceinline _TInput* CopyOutWeights() const;
#ifdef __CUDACC__
                __device__ __forceinline _TInput* CopyOutWeights() const;
#endif
                __host__ __device__ __forceinline void CopyOutWeights(_TInput* Weights) const;
                __host__ __device__ __forceinline void CopyInWeights(_TInput* Weights);
                __host__ __device__ __forceinline _TInput GetWeight(size_t Index) const;
                __host__ __device__ __forceinline void SetWeight(size_t Index, _TInput Weight);

                __host__ __device__ __forceinline _TOutput GetBias() const;
                __host__ __device__ __forceinline void SetBias(_TOutput Bias);

                __host__ __device__ __forceinline _TOutput Run(_TInput Input) const;
                __host__ __device__ __forceinline uint64_t RunG(uint64_t Input) const;

                __host__ __device__ __forceinline void CopyTo(MLPBL<_TInput, _TOutput> Other) const;
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> Clone() const;

                __host__ __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ __forceinline void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                __forceinline size_t SerializedSize() const;
                __forceinline void Serialize(void*& Data) const;
                static __forceinline MLPBL<_TInput, _TOutput> Deserialize(const void*& Data);
                static __forceinline void Deserialize(const void*& Data, void* ObjMem);
            private:
                _TInput* weights;
                _TOutput* bias;
            };

            using mlpbl8T8_t = MLPBL<uint8_t, uint8_t>;
            using mlpbl8T16_t = MLPBL<uint8_t, uint16_t>;
            using mlpbl8T32_t = MLPBL<uint8_t, uint32_t>;
            using mlpbl8T64_t = MLPBL<uint8_t, uint64_t>;
            using mlpbl16T8_t = MLPBL<uint16_t, uint8_t>;
            using mlpbl16T16_t = MLPBL<uint16_t, uint16_t>;
            using mlpbl16T32_t = MLPBL<uint16_t, uint32_t>;
            using mlpbl16T64_t = MLPBL<uint16_t, uint64_t>;
            using mlpbl32T8_t = MLPBL<uint32_t, uint8_t>;
            using mlpbl32T16_t = MLPBL<uint32_t, uint16_t>;
            using mlpbl32T32_t = MLPBL<uint32_t, uint32_t>;
            using mlpbl32T64_t = MLPBL<uint32_t, uint64_t>;
            using mlpbl64T8_t = MLPBL<uint64_t, uint8_t>;
            using mlpbl64T16_t = MLPBL<uint64_t, uint16_t>;
            using mlpbl64T32_t = MLPBL<uint64_t, uint32_t>;
            using mlpbl64T64_t = MLPBL<uint64_t, uint64_t>;
        }
    }
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::MLPBL() {
#ifdef __CUDA_ARCH__
    weights = new _TInput[sizeof(_TOutput) << 3];
    bias = new _TOutput;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(_TInput) * (sizeof(_TOutput) << 3)));
    ThrowIfBad(cudaMalloc(&bias, sizeof(_TOutput)));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::MLPBL(_TInput* Weights, _TOutput* Bias) {
    weights = Weights;
    bias = Bias;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CloneFrom(_TInput* Weights, _TOutput* Bias) -> MLPBL<_TInput, _TOutput> {
#ifdef __CUDA_ARCH__
    _TInput* weights = new _TInput[sizeof(_TOutput) << 3];
    _TOutput* bias = new _TOutput;
    deviceMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
    *bias = *Bias;
    return MLPBL(weights, bias);
#else
    _TInput* weights;
    _TOutput* bias;
    ThrowIfBad(cudaMalloc(&weights, sizeof(_TInput) * (sizeof(_TOutput) << 3)));
    ThrowIfBad(cudaMalloc(&bias, sizeof(_TOutput)));
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDefault));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(_TOutput), cudaMemcpyDefault));
    return MLPBL(weights, bias);
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Dispose() {
#ifdef __CUDA_ARCH__
    delete[] weights;
    delete bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Weights() const {
    return weights;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Weight(size_t Index) const {
    return &weights[Index];
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TOutput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Bias() const {
    return bias;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <bool _CopyToHost>
__host__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyOutWeights() const {
    if (_CopyToHost) {
        _TInput* output = new _TInput[sizeof(_TOutput) << 3];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        _TInput* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(_TInput) * (sizeof(_TOutput) << 3)));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToDevice));
        return output;
    }
}
#ifdef __CUDACC__
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__device__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyOutWeights() const {
    _TInput* output = new _TInput[sizeof(_TOutput) << 3];
    deviceMemcpy(output, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
    return output;
}
#endif
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyOutWeights(_TInput* Weights) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
#else
    cudaMemcpy(Weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDefault);
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyInWeights(_TInput* Weights) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
#else
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDefault));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TInput BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::GetWeight(size_t Index) const {
#ifdef __CUDA_ARCH__
    return weights[Index];
#else
    _TInput output;
    ThrowIfBad(cudaMemcpy(&output, weights + Index, sizeof(_TInput), cudaMemcpyDeviceToHost));
    return output;
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::SetWeight(size_t Index, _TInput Weight) {
#ifdef __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(weights + Index, &Weight, sizeof(_TInput), cudaMemcpyHostToDevice));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TOutput BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::GetBias() const {
#ifdef __CUDA_ARCH__
    return *bias;
#else
    _TOutput output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost));
    return output;
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::SetBias(_TOutput Bias) {
#ifdef __CUDA_ARCH__
    * bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(_TOutput), cudaMemcpyHostToDevice));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TOutput BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Run(_TInput Input) const {
#ifdef __CUDA_ARCH__
    _TOutput o = 0;
    for (size_t i = 0; i < (sizeof(_TOutput) << 3); ++i) {
        if (Input & weights[i]) o |= ((_TOutput)1) << i;
    }
    return o ^ (*bias);
#else
    _TInput hWeights[sizeof(_TOutput) << 3];
    _TOutput o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < (sizeof(_TOutput) << 3); ++i) {
        if (Input & hWeights[i]) o |= ((_TOutput)1) << i;
    }
    return o;
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline uint64_t BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RunG(uint64_t Input) const {
    return (uint64_t)Run((_TInput)Input);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyTo(MLPBL<_TInput, _TOutput> Other) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
    deviceMemcpy(Other.bias, bias, sizeof(_TOutput));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(_TOutput), cudaMemcpyDeviceToDevice));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Clone() const {
    MLPBL<_TInput, _TOutput> n;
    CopyTo(n);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    RandomizeArray(Span<_TInput>(weights, sizeof(_TOutput) << 3), WeightsFlipProb, RNG);
    RandomizeArray(Span<_TOutput>(bias, 1), BiasFlipProb, RNG);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    MLPBL<_TInput, _TOutput> n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    details::applyTargetFlipsOnArray(weights, sizeof(_TOutput) << 3, WeightsEachFlipProb, RNG);
    RandomizeArray(Span<_TOutput>(bias, 1), BiasFlipProb, RNG);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    MLPBL<_TInput, _TOutput> n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) {
    for (uint32_t i = 0; i < (sizeof(_TOutput) << 3); ++i) {
        if (RNG() < WeightsMutationProb) {
            SetWeight(i, (_TInput)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, RNG));
        }
    }
    if (RNG() < WeightsMutationProb) {
        SetBias((_TOutput)BrendanCUDA::Random::Get64Bits(BiasProbOf1, RNG));
    }
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const {
    MLPBL<_TInput, _TOutput> n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline size_t BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::SerializedSize() const {
    return sizeof(_TInput) * (sizeof(_TOutput) << 3) + sizeof(_TOutput);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Serialize(void*& Data) const {
    if constexpr (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost);
        Data = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        cudaMemcpy(Data, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost);
        Data = ((_TOutput*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost);
        void* nd = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        BSerializer::ToFromLittleEndian((_TInput*)Data, (_TInput*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost);
        _TOutput& bref = *(_TOutput*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((_TOutput*)Data) + 1;
    }
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Deserialize(const void*& Data) {
    if constexpr (std::endian::native == std::endian::little) {
        MLPBL<_TInput, _TOutput> mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyHostToDevice);
        Data = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        cudaMemcpy(mlpbl.bias, Data, sizeof(_TOutput), cudaMemcpyHostToDevice);
        Data = ((_TOutput*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyHostToDevice);
        void* nd = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        BSerializer::ToFromLittleEndian((_TInput*)Data, (_TInput*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(_TOutput), cudaMemcpyHostToDevice);
        _TOutput& bref = *(_TOutput*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((_TOutput*)Data) + 1;
        return mlpbl;
    }
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Deserialize(const void*& Data, void* ObjMem) {
    new (ObjMem) MLPBL(Deserialize(Data));
}