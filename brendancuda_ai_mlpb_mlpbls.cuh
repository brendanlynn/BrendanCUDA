#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "brendancuda_devicecopy.cuh"
#include "brendancuda_random_rngfunc.cuh"
#include "brendancuda_ai.cuh"

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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL8T8 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL8T16 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL8T32 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL8T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL8T64 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL16T8 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL16T16 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL16T32 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL16T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL16T64 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL32T8 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL32T16 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL32T32 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL32T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL32T64 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T8 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T8 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T8 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL64T8 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T16 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T16 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T16 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL64T16 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T32 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T32 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T32 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL64T32 Deserialize(std::basic_istream<char>& Stream);
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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T64 ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T64 ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPBL64T64 ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ MLPBL64T64 Deserialize(std::basic_istream<char>& Stream);
            private:
                uint64_t* weights;
                uint64_t* bias;
            };
        }
    }
}