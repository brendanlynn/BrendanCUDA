#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <bit>
#include "BSerializer/Serializer.h"
#include "brendancuda_rand_anyrng.h"
#include "brendancuda_ai.h"

namespace BrendanCUDA {
    namespace AI {
        namespace MLPB {
            class MLPBL8T8 final {
            public:
                __host__ __device__ MLPBL8T8();
                __host__ MLPBL8T8(uint8_t* Weights, uint8_t* Bias, bool CopyFromHost);
                __device__ MLPBL8T8(uint8_t* Weights, uint8_t* Bias);
                __host__ __device__ MLPBL8T8(uint8_t* Weights, uint8_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint8_t* Weights() const;
                __host__ __device__ uint8_t* Weight(size_t Index) const;
                __host__ __device__ uint8_t* Bias() const;
                __host__ uint8_t* GetWeights(bool CopyToHost) const;
                __device__ uint8_t* GetWeights() const;
                __host__ void SetWeights(uint8_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint8_t* Weights);
                __host__ __device__ uint8_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint8_t Weight);
                __host__ __device__ uint8_t GetBias() const;
                __host__ __device__ void SetBias(uint8_t Bias);
                __host__ __device__ uint8_t Run(uint8_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL8T8 Other) const;

                __host__ __device__ MLPBL8T8 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL8T8 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL8T8& Value);
            private:
                uint8_t* weights;
                uint8_t* bias;
            };
            class MLPBL8T16 final {
            public:
                __host__ __device__ MLPBL8T16();
                __host__ MLPBL8T16(uint8_t* Weights, uint16_t* Bias, bool CopyFromHost);
                __device__ MLPBL8T16(uint8_t* Weights, uint16_t* Bias);
                __host__ __device__ MLPBL8T16(uint8_t* Weights, uint16_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint8_t* Weights() const;
                __host__ __device__ uint8_t* Weight(size_t Index) const;
                __host__ __device__ uint16_t* Bias() const;
                __host__ uint8_t* GetWeights(bool CopyToHost) const;
                __device__ uint8_t* GetWeights() const;
                __host__ void SetWeights(uint8_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint8_t* Weights);
                __host__ __device__ uint8_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint8_t Weight);
                __host__ __device__ uint16_t GetBias() const;
                __host__ __device__ void SetBias(uint16_t Bias);
                __host__ __device__ uint16_t Run(uint8_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL8T16 Other) const;

                __host__ __device__ MLPBL8T16 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL8T16 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL8T16& Value);
            private:
                uint8_t* weights;
                uint16_t* bias;
            };
            class MLPBL8T32 final {
            public:
                __host__ __device__ MLPBL8T32();
                __host__ MLPBL8T32(uint8_t* Weights, uint32_t* Bias, bool CopyFromHost);
                __device__ MLPBL8T32(uint8_t* Weights, uint32_t* Bias);
                __host__ __device__ MLPBL8T32(uint8_t* Weights, uint32_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint8_t* Weights() const;
                __host__ __device__ uint8_t* Weight(size_t Index) const;
                __host__ __device__ uint32_t* Bias() const;
                __host__ uint8_t* GetWeights(bool CopyToHost) const;
                __device__ uint8_t* GetWeights() const;
                __host__ void SetWeights(uint8_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint8_t* Weights);
                __host__ __device__ uint8_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint8_t Weight);
                __host__ __device__ uint32_t GetBias() const;
                __host__ __device__ void SetBias(uint32_t Bias);
                __host__ __device__ uint32_t Run(uint8_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL8T32 Other) const;

                __host__ __device__ MLPBL8T32 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL8T32 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL8T32& Value);
            private:
                uint8_t* weights;
                uint32_t* bias;
            };
            class MLPBL8T64 final {
            public:
                __host__ __device__ MLPBL8T64();
                __host__ MLPBL8T64(uint8_t* Weights, uint64_t* Bias, bool CopyFromHost);
                __device__ MLPBL8T64(uint8_t* Weights, uint64_t* Bias);
                __host__ __device__ MLPBL8T64(uint8_t* Weights, uint64_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint8_t* Weights() const;
                __host__ __device__ uint8_t* Weight(size_t Index) const;
                __host__ __device__ uint64_t* Bias() const;
                __host__ uint8_t* GetWeights(bool CopyToHost) const;
                __device__ uint8_t* GetWeights() const;
                __host__ void SetWeights(uint8_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint8_t* Weights);
                __host__ __device__ uint8_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint8_t Weight);
                __host__ __device__ uint64_t GetBias() const;
                __host__ __device__ void SetBias(uint64_t Bias);
                __host__ __device__ uint64_t Run(uint8_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL8T64 Other) const;

                __host__ __device__ MLPBL8T64 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL8T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL8T64 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL8T64& Value);
            private:
                uint8_t* weights;
                uint64_t* bias;
            };
            class MLPBL16T8 final {
            public:
                __host__ __device__ MLPBL16T8();
                __host__ MLPBL16T8(uint16_t* Weights, uint8_t* Bias, bool CopyFromHost);
                __device__ MLPBL16T8(uint16_t* Weights, uint8_t* Bias);
                __host__ __device__ MLPBL16T8(uint16_t* Weights, uint8_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint16_t* Weights() const;
                __host__ __device__ uint16_t* Weight(size_t Index) const;
                __host__ __device__ uint8_t* Bias() const;
                __host__ uint16_t* GetWeights(bool CopyToHost) const;
                __device__ uint16_t* GetWeights() const;
                __host__ void SetWeights(uint16_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint16_t* Weights);
                __host__ __device__ uint16_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint16_t Weight);
                __host__ __device__ uint8_t GetBias() const;
                __host__ __device__ void SetBias(uint8_t Bias);
                __host__ __device__ uint8_t Run(uint16_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL16T8 Other) const;

                __host__ __device__ MLPBL16T8 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL16T8 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL16T8& Value);
            private:
                uint16_t* weights;
                uint8_t* bias;
            };
            class MLPBL16T16 final {
            public:
                __host__ __device__ MLPBL16T16();
                __host__ MLPBL16T16(uint16_t* Weights, uint16_t* Bias, bool CopyFromHost);
                __device__ MLPBL16T16(uint16_t* Weights, uint16_t* Bias);
                __host__ __device__ MLPBL16T16(uint16_t* Weights, uint16_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint16_t* Weights() const;
                __host__ __device__ uint16_t* Weight(size_t Index) const;
                __host__ __device__ uint16_t* Bias() const;
                __host__ uint16_t* GetWeights(bool CopyToHost) const;
                __device__ uint16_t* GetWeights() const;
                __host__ void SetWeights(uint16_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint16_t* Weights);
                __host__ __device__ uint16_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint16_t Weight);
                __host__ __device__ uint16_t GetBias() const;
                __host__ __device__ void SetBias(uint16_t Bias);
                __host__ __device__ uint16_t Run(uint16_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL16T16 Other) const;

                __host__ __device__ MLPBL16T16 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL16T16 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL16T16& Value);
            private:
                uint16_t* weights;
                uint16_t* bias;
            };
            class MLPBL16T32 final {
            public:
                __host__ __device__ MLPBL16T32();
                __host__ MLPBL16T32(uint16_t* Weights, uint32_t* Bias, bool CopyFromHost);
                __device__ MLPBL16T32(uint16_t* Weights, uint32_t* Bias);
                __host__ __device__ MLPBL16T32(uint16_t* Weights, uint32_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint16_t* Weights() const;
                __host__ __device__ uint16_t* Weight(size_t Index) const;
                __host__ __device__ uint32_t* Bias() const;
                __host__ uint16_t* GetWeights(bool CopyToHost) const;
                __device__ uint16_t* GetWeights() const;
                __host__ void SetWeights(uint16_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint16_t* Weights);
                __host__ __device__ uint16_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint16_t Weight);
                __host__ __device__ uint32_t GetBias() const;
                __host__ __device__ void SetBias(uint32_t Bias);
                __host__ __device__ uint32_t Run(uint16_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL16T32 Other) const;

                __host__ __device__ MLPBL16T32 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL16T32 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL16T32& Value);
            private:
                uint16_t* weights;
                uint32_t* bias;
            };
            class MLPBL16T64 final {
            public:
                __host__ __device__ MLPBL16T64();
                __host__ MLPBL16T64(uint16_t* Weights, uint64_t* Bias, bool CopyFromHost);
                __device__ MLPBL16T64(uint16_t* Weights, uint64_t* Bias);
                __host__ __device__ MLPBL16T64(uint16_t* Weights, uint64_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint16_t* Weights() const;
                __host__ __device__ uint16_t* Weight(size_t Index) const;
                __host__ __device__ uint64_t* Bias() const;
                __host__ uint16_t* GetWeights(bool CopyToHost) const;
                __device__ uint16_t* GetWeights() const;
                __host__ void SetWeights(uint16_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint16_t* Weights);
                __host__ __device__ uint16_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint16_t Weight);
                __host__ __device__ uint64_t GetBias() const;
                __host__ __device__ void SetBias(uint64_t Bias);
                __host__ __device__ uint64_t Run(uint16_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL16T64 Other) const;

                __host__ __device__ MLPBL16T64 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL16T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL16T64 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL16T64& Value);
            private:
                uint16_t* weights;
                uint64_t* bias;
            };
            class MLPBL32T8 final {
            public:
                __host__ __device__ MLPBL32T8();
                __host__ MLPBL32T8(uint32_t* Weights, uint8_t* Bias, bool CopyFromHost);
                __device__ MLPBL32T8(uint32_t* Weights, uint8_t* Bias);
                __host__ __device__ MLPBL32T8(uint32_t* Weights, uint8_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint32_t* Weights() const;
                __host__ __device__ uint32_t* Weight(size_t Index) const;
                __host__ __device__ uint8_t* Bias() const;
                __host__ uint32_t* GetWeights(bool CopyToHost) const;
                __device__ uint32_t* GetWeights() const;
                __host__ void SetWeights(uint32_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint32_t* Weights);
                __host__ __device__ uint32_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint32_t Weight);
                __host__ __device__ uint8_t GetBias() const;
                __host__ __device__ void SetBias(uint8_t Bias);
                __host__ __device__ uint8_t Run(uint32_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL32T8 Other) const;

                __host__ __device__ MLPBL32T8 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL32T8 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL32T8& Value);
            private:
                uint32_t* weights;
                uint8_t* bias;
            };
            class MLPBL32T16 final {
            public:
                __host__ __device__ MLPBL32T16();
                __host__ MLPBL32T16(uint32_t* Weights, uint16_t* Bias, bool CopyFromHost);
                __device__ MLPBL32T16(uint32_t* Weights, uint16_t* Bias);
                __host__ __device__ MLPBL32T16(uint32_t* Weights, uint16_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint32_t* Weights() const;
                __host__ __device__ uint32_t* Weight(size_t Index) const;
                __host__ __device__ uint16_t* Bias() const;
                __host__ uint32_t* GetWeights(bool CopyToHost) const;
                __device__ uint32_t* GetWeights() const;
                __host__ void SetWeights(uint32_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint32_t* Weights);
                __host__ __device__ uint32_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint32_t Weight);
                __host__ __device__ uint16_t GetBias() const;
                __host__ __device__ void SetBias(uint16_t Bias);
                __host__ __device__ uint16_t Run(uint32_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL32T16 Other) const;

                __host__ __device__ MLPBL32T16 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL32T16 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL32T16& Value);
            private:
                uint32_t* weights;
                uint16_t* bias;
            };
            class MLPBL32T32 final {
            public:
                __host__ __device__ MLPBL32T32();
                __host__ MLPBL32T32(uint32_t* Weights, uint32_t* Bias, bool CopyFromHost);
                __device__ MLPBL32T32(uint32_t* Weights, uint32_t* Bias);
                __host__ __device__ MLPBL32T32(uint32_t* Weights, uint32_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint32_t* Weights() const;
                __host__ __device__ uint32_t* Weight(size_t Index) const;
                __host__ __device__ uint32_t* Bias() const;
                __host__ uint32_t* GetWeights(bool CopyToHost) const;
                __device__ uint32_t* GetWeights() const;
                __host__ void SetWeights(uint32_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint32_t* Weights);
                __host__ __device__ uint32_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint32_t Weight);
                __host__ __device__ uint32_t GetBias() const;
                __host__ __device__ void SetBias(uint32_t Bias);
                __host__ __device__ uint32_t Run(uint32_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL32T32 Other) const;

                __host__ __device__ MLPBL32T32 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL32T32 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL32T32& Value);
            private:
                uint32_t* weights;
                uint32_t* bias;
            };
            class MLPBL32T64 final {
            public:
                __host__ __device__ MLPBL32T64();
                __host__ MLPBL32T64(uint32_t* Weights, uint64_t* Bias, bool CopyFromHost);
                __device__ MLPBL32T64(uint32_t* Weights, uint64_t* Bias);
                __host__ __device__ MLPBL32T64(uint32_t* Weights, uint64_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint32_t* Weights() const;
                __host__ __device__ uint32_t* Weight(size_t Index) const;
                __host__ __device__ uint64_t* Bias() const;
                __host__ uint32_t* GetWeights(bool CopyToHost) const;
                __device__ uint32_t* GetWeights() const;
                __host__ void SetWeights(uint32_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint32_t* Weights);
                __host__ __device__ uint32_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint32_t Weight);
                __host__ __device__ uint64_t GetBias() const;
                __host__ __device__ void SetBias(uint64_t Bias);
                __host__ __device__ uint64_t Run(uint32_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL32T64 Other) const;

                __host__ __device__ MLPBL32T64 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL32T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL32T64 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL32T64& Value);
            private:
                uint32_t* weights;
                uint64_t* bias;
            };
            class MLPBL64T8 final {
            public:
                __host__ __device__ MLPBL64T8();
                __host__ MLPBL64T8(uint64_t* Weights, uint8_t* Bias, bool CopyFromHost);
                __device__ MLPBL64T8(uint64_t* Weights, uint8_t* Bias);
                __host__ __device__ MLPBL64T8(uint64_t* Weights, uint8_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint64_t* Weights() const;
                __host__ __device__ uint64_t* Weight(size_t Index) const;
                __host__ __device__ uint8_t* Bias() const;
                __host__ uint64_t* GetWeights(bool CopyToHost) const;
                __device__ uint64_t* GetWeights() const;
                __host__ void SetWeights(uint64_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint64_t* Weights);
                __host__ __device__ uint64_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint64_t Weight);
                __host__ __device__ uint8_t GetBias() const;
                __host__ __device__ void SetBias(uint8_t Bias);
                __host__ __device__ uint8_t Run(uint64_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL64T8 Other) const;

                __host__ __device__ MLPBL64T8 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL64T8 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL64T8& Value);
            private:
                uint64_t* weights;
                uint8_t* bias;
            };
            class MLPBL64T16 final {
            public:
                __host__ __device__ MLPBL64T16();
                __host__ MLPBL64T16(uint64_t* Weights, uint16_t* Bias, bool CopyFromHost);
                __device__ MLPBL64T16(uint64_t* Weights, uint16_t* Bias);
                __host__ __device__ MLPBL64T16(uint64_t* Weights, uint16_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint64_t* Weights() const;
                __host__ __device__ uint64_t* Weight(size_t Index) const;
                __host__ __device__ uint16_t* Bias() const;
                __host__ uint64_t* GetWeights(bool CopyToHost) const;
                __device__ uint64_t* GetWeights() const;
                __host__ void SetWeights(uint64_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint64_t* Weights);
                __host__ __device__ uint64_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint64_t Weight);
                __host__ __device__ uint16_t GetBias() const;
                __host__ __device__ void SetBias(uint16_t Bias);
                __host__ __device__ uint16_t Run(uint64_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL64T16 Other) const;

                __host__ __device__ MLPBL64T16 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL64T16 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL64T16& Value);
            private:
                uint64_t* weights;
                uint16_t* bias;
            };
            class MLPBL64T32 final {
            public:
                __host__ __device__ MLPBL64T32();
                __host__ MLPBL64T32(uint64_t* Weights, uint32_t* Bias, bool CopyFromHost);
                __device__ MLPBL64T32(uint64_t* Weights, uint32_t* Bias);
                __host__ __device__ MLPBL64T32(uint64_t* Weights, uint32_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint64_t* Weights() const;
                __host__ __device__ uint64_t* Weight(size_t Index) const;
                __host__ __device__ uint32_t* Bias() const;
                __host__ uint64_t* GetWeights(bool CopyToHost) const;
                __device__ uint64_t* GetWeights() const;
                __host__ void SetWeights(uint64_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint64_t* Weights);
                __host__ __device__ uint64_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint64_t Weight);
                __host__ __device__ uint32_t GetBias() const;
                __host__ __device__ void SetBias(uint32_t Bias);
                __host__ __device__ uint32_t Run(uint64_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL64T32 Other) const;

                __host__ __device__ MLPBL64T32 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL64T32 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL64T32& Value);
            private:
                uint64_t* weights;
                uint32_t* bias;
            };
            class MLPBL64T64 final {
            public:
                __host__ __device__ MLPBL64T64();
                __host__ MLPBL64T64(uint64_t* Weights, uint64_t* Bias, bool CopyFromHost);
                __device__ MLPBL64T64(uint64_t* Weights, uint64_t* Bias);
                __host__ __device__ MLPBL64T64(uint64_t* Weights, uint64_t Bias);
                __host__ __device__ void Dispose();
                __host__ __device__ uint64_t* Weights() const;
                __host__ __device__ uint64_t* Weight(size_t Index) const;
                __host__ __device__ uint64_t* Bias() const;
                __host__ uint64_t* GetWeights(bool CopyToHost) const;
                __device__ uint64_t* GetWeights() const;
                __host__ void SetWeights(uint64_t* Weights, bool CopyFromHost);
                __device__ void SetWeights(uint64_t* Weights);
                __host__ __device__ uint64_t GetWeight(size_t Index) const;
                __host__ __device__ void SetWeight(size_t Index, uint64_t Weight);
                __host__ __device__ uint64_t GetBias() const;
                __host__ __device__ void SetBias(uint64_t Bias);
                __host__ __device__ uint64_t Run(uint64_t Input) const;
                __host__ __device__ uint64_t RunG(uint64_t Input) const;
                __host__ __device__ void CopyTo(MLPBL64T64 Other) const;

                __host__ __device__ MLPBL64T64 Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL64T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL64T64 Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPBL64T64& Value);
            private:
                uint64_t* weights;
                uint64_t* bias;
            };
        }
    }
}

size_t BrendanCUDA::AI::MLPB::MLPBL8T8::SerializedSize() const {
    return sizeof(uint8_t) * 8 + sizeof(uint8_t);
}
void BrendanCUDA::AI::MLPB::MLPBL8T8::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 8, cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 8;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 8, cudaMemcpyDeviceToHost);
        void* nd = ((uint8_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL8T8 BrendanCUDA::AI::MLPB::MLPBL8T8::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL8T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 8, cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 8;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL8T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 8, cudaMemcpyHostToDevice);
        void* nd = ((uint8_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL8T8::Deserialize(const void*& Data, MLPBL8T8& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL8T16::SerializedSize() const {
    return sizeof(uint8_t) * 16 + sizeof(uint16_t);
}
void BrendanCUDA::AI::MLPB::MLPBL8T16::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 16, cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 16;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 16, cudaMemcpyDeviceToHost);
        void* nd = ((uint8_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL8T16 BrendanCUDA::AI::MLPB::MLPBL8T16::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL8T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 16, cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 16;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL8T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 16, cudaMemcpyHostToDevice);
        void* nd = ((uint8_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL8T16::Deserialize(const void*& Data, MLPBL8T16& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL8T32::SerializedSize() const {
    return sizeof(uint8_t) * 32 + sizeof(uint32_t);
}
void BrendanCUDA::AI::MLPB::MLPBL8T32::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 32, cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 32;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 32, cudaMemcpyDeviceToHost);
        void* nd = ((uint8_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL8T32 BrendanCUDA::AI::MLPB::MLPBL8T32::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL8T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 32, cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 32;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL8T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 32, cudaMemcpyHostToDevice);
        void* nd = ((uint8_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL8T32::Deserialize(const void*& Data, MLPBL8T32& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL8T64::SerializedSize() const {
    return sizeof(uint8_t) * 64 + sizeof(uint64_t);
}
void BrendanCUDA::AI::MLPB::MLPBL8T64::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 64, cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 64;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint8_t) * 64, cudaMemcpyDeviceToHost);
        void* nd = ((uint8_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL8T64 BrendanCUDA::AI::MLPB::MLPBL8T64::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL8T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 64, cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 64;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL8T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint8_t) * 64, cudaMemcpyHostToDevice);
        void* nd = ((uint8_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint8_t*)Data, (uint8_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL8T64::Deserialize(const void*& Data, MLPBL8T64& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL16T8::SerializedSize() const {
    return sizeof(uint16_t) * 8 + sizeof(uint8_t);
}
void BrendanCUDA::AI::MLPB::MLPBL16T8::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 8, cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 8;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 8, cudaMemcpyDeviceToHost);
        void* nd = ((uint16_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL16T8 BrendanCUDA::AI::MLPB::MLPBL16T8::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL16T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 8, cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 8;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL16T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 8, cudaMemcpyHostToDevice);
        void* nd = ((uint16_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL16T8::Deserialize(const void*& Data, MLPBL16T8& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL16T16::SerializedSize() const {
    return sizeof(uint16_t) * 16 + sizeof(uint16_t);
}
void BrendanCUDA::AI::MLPB::MLPBL16T16::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 16, cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 16;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 16, cudaMemcpyDeviceToHost);
        void* nd = ((uint16_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL16T16 BrendanCUDA::AI::MLPB::MLPBL16T16::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL16T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 16, cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 16;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL16T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 16, cudaMemcpyHostToDevice);
        void* nd = ((uint16_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL16T16::Deserialize(const void*& Data, MLPBL16T16& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL16T32::SerializedSize() const {
    return sizeof(uint16_t) * 32 + sizeof(uint32_t);
}
void BrendanCUDA::AI::MLPB::MLPBL16T32::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 32, cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 32;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 32, cudaMemcpyDeviceToHost);
        void* nd = ((uint16_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL16T32 BrendanCUDA::AI::MLPB::MLPBL16T32::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL16T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 32, cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 32;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL16T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 32, cudaMemcpyHostToDevice);
        void* nd = ((uint16_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL16T32::Deserialize(const void*& Data, MLPBL16T32& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL16T64::SerializedSize() const {
    return sizeof(uint16_t) * 64 + sizeof(uint64_t);
}
void BrendanCUDA::AI::MLPB::MLPBL16T64::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 64, cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 64;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint16_t) * 64, cudaMemcpyDeviceToHost);
        void* nd = ((uint16_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL16T64 BrendanCUDA::AI::MLPB::MLPBL16T64::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL16T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 64, cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 64;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL16T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint16_t) * 64, cudaMemcpyHostToDevice);
        void* nd = ((uint16_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint16_t*)Data, (uint16_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL16T64::Deserialize(const void*& Data, MLPBL16T64& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL32T8::SerializedSize() const {
    return sizeof(uint32_t) * 8 + sizeof(uint8_t);
}
void BrendanCUDA::AI::MLPB::MLPBL32T8::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 8;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost);
        void* nd = ((uint32_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL32T8 BrendanCUDA::AI::MLPB::MLPBL32T8::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL32T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 8, cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 8;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL32T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 8, cudaMemcpyHostToDevice);
        void* nd = ((uint32_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL32T8::Deserialize(const void*& Data, MLPBL32T8& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL32T16::SerializedSize() const {
    return sizeof(uint32_t) * 16 + sizeof(uint16_t);
}
void BrendanCUDA::AI::MLPB::MLPBL32T16::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 16;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost);
        void* nd = ((uint32_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL32T16 BrendanCUDA::AI::MLPB::MLPBL32T16::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL32T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 16, cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 16;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL32T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 16, cudaMemcpyHostToDevice);
        void* nd = ((uint32_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL32T16::Deserialize(const void*& Data, MLPBL32T16& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL32T32::SerializedSize() const {
    return sizeof(uint32_t) * 32 + sizeof(uint32_t);
}
void BrendanCUDA::AI::MLPB::MLPBL32T32::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 32, cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 32;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 32, cudaMemcpyDeviceToHost);
        void* nd = ((uint32_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL32T32 BrendanCUDA::AI::MLPB::MLPBL32T32::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL32T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 32;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL32T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice);
        void* nd = ((uint32_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL32T32::Deserialize(const void*& Data, MLPBL32T32& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL32T64::SerializedSize() const {
    return sizeof(uint32_t) * 64 + sizeof(uint64_t);
}
void BrendanCUDA::AI::MLPB::MLPBL32T64::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 64, cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 64;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint32_t) * 64, cudaMemcpyDeviceToHost);
        void* nd = ((uint32_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL32T64 BrendanCUDA::AI::MLPB::MLPBL32T64::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL32T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 64, cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 64;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL32T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint32_t) * 64, cudaMemcpyHostToDevice);
        void* nd = ((uint32_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint32_t*)Data, (uint32_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL32T64::Deserialize(const void*& Data, MLPBL32T64& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL64T8::SerializedSize() const {
    return sizeof(uint64_t) * 8 + sizeof(uint8_t);
}
void BrendanCUDA::AI::MLPB::MLPBL64T8::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 8, cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 8;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        Data = ((uint8_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 8, cudaMemcpyDeviceToHost);
        void* nd = ((uint64_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL64T8 BrendanCUDA::AI::MLPB::MLPBL64T8::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL64T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 8, cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 8;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL64T8 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 8, cudaMemcpyHostToDevice);
        void* nd = ((uint64_t*)Data) + 8;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint8_t), cudaMemcpyHostToDevice);
        uint8_t& bref = *(uint8_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint8_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL64T8::Deserialize(const void*& Data, MLPBL64T8& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL64T16::SerializedSize() const {
    return sizeof(uint64_t) * 16 + sizeof(uint16_t);
}
void BrendanCUDA::AI::MLPB::MLPBL64T16::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 16, cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 16;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        Data = ((uint16_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 16, cudaMemcpyDeviceToHost);
        void* nd = ((uint64_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL64T16 BrendanCUDA::AI::MLPB::MLPBL64T16::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL64T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 16, cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 16;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL64T16 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 16, cudaMemcpyHostToDevice);
        void* nd = ((uint64_t*)Data) + 16;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint16_t), cudaMemcpyHostToDevice);
        uint16_t& bref = *(uint16_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint16_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL64T16::Deserialize(const void*& Data, MLPBL64T16& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL64T32::SerializedSize() const {
    return sizeof(uint64_t) * 32 + sizeof(uint32_t);
}
void BrendanCUDA::AI::MLPB::MLPBL64T32::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 32, cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 32;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        Data = ((uint32_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 32, cudaMemcpyDeviceToHost);
        void* nd = ((uint64_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL64T32 BrendanCUDA::AI::MLPB::MLPBL64T32::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL64T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 32, cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 32;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL64T32 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 32, cudaMemcpyHostToDevice);
        void* nd = ((uint64_t*)Data) + 32;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint32_t), cudaMemcpyHostToDevice);
        uint32_t& bref = *(uint32_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint32_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL64T32::Deserialize(const void*& Data, MLPBL64T32& Value) {
    Value = Deserialize(Data);
}
size_t BrendanCUDA::AI::MLPB::MLPBL64T64::SerializedSize() const {
    return sizeof(uint64_t) * 64 + sizeof(uint64_t);
}
void BrendanCUDA::AI::MLPB::MLPBL64T64::Serialize(void*& Data) const {
    if (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 64, cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 64;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        Data = ((uint64_t*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(uint64_t) * 64, cudaMemcpyDeviceToHost);
        void* nd = ((uint64_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
    }
}
BrendanCUDA::AI::MLPB::MLPBL64T64 BrendanCUDA::AI::MLPB::MLPBL64T64::Deserialize(const void*& Data) {
    if (std::endian::native == std::endian::little) {
        MLPBL64T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 64, cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 64;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL64T64 mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(uint64_t) * 64, cudaMemcpyHostToDevice);
        void* nd = ((uint64_t*)Data) + 64;
        BSerializer::ToFromLittleEndian((uint64_t*)Data, (uint64_t*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(uint64_t), cudaMemcpyHostToDevice);
        uint64_t& bref = *(uint64_t*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((uint64_t*)Data) + 1;
        return mlpbl;
    }
}
void BrendanCUDA::AI::MLPB::MLPBL64T64::Deserialize(const void*& Data, MLPBL64T64& Value) {
    Value = Deserialize(Data);
}
