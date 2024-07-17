#pragma once

#include <memory>
#include "brendancuda_ai_mlpb_mlpbl.h"
#include "brendancuda_rand_anyrng.h"
#include "BSerializer/Serializer.h"

namespace BrendanCUDA {
    namespace AI {
        namespace MLPB {
            class MLPB final {
            public:
                __host__ __device__ MLPB(size_t LayerCount);
                __host__ MLPB(MLPBL* Layers, size_t LayerCount, bool CopyFromHost);
                __device__ MLPB(MLPBL* Layers, size_t LayerCount);

                __host__ __device__ void Dispose();

                __host__ MLPBL* GetLayers(bool CopyToHost) const;
                __device__ MLPBL* GetLayers() const;
                __host__ void SetLayers(MLPBL* Layers, bool CopyFromHost);
                __device__ void SetLayers(MLPBL* Layers);

                __host__ __device__ MLPBL GetLayer(size_t LayerIndex) const;
                __host__ __device__ void SetLayer(size_t LayerIndex, MLPBL Layer);

                __host__ __device__ MLPBL* Layers() const;
                __host__ __device__ MLPBL* Layer(size_t LayerIndex) const;
                __host__ __device__ size_t LayerCount() const;

                __host__ __device__ uint64_t Run(uint64_t Input) const;

                __host__ __device__ MLPB operator+(const MLPB Value);
                __host__ __device__ MLPB operator+(const MLPBL Value);

                __host__ __device__ MLPB Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPB ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPB ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPB ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPB Deserialize(const void*& Data);
                static void Deserialize(const void*& Data, MLPB& Value);
            private:
                MLPBL* layers;
                size_t layerCount;
            };
        }
    }
}

size_t BrendanCUDA::AI::MLPB::MLPB::SerializedSize() const {
    size_t t = sizeof(size_t);
    for (size_t i = 0; i < layerCount; ++i) {
        t += GetLayer(i).SerializedSize();
    }
    return t;
}
void BrendanCUDA::AI::MLPB::MLPB::Serialize(void*& Data) const {
    BSerializer::Serialize(Data, layerCount);
    for (size_t i = 0; i < layerCount; ++i) {
        GetLayer(i).Serialize(Data);
    }
}
BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::Deserialize(const void*& Data) {
    MLPB mlpb(BSerializer::Deserialize<size_t>(Data));
    for (size_t i = 0; i < mlpb.layerCount; ++i) {
        mlpb.SetLayer(i, MLPBL::Deserialize(Data));
    }
    return mlpb;
}
void BrendanCUDA::AI::MLPB::MLPB::Deserialize(const void*& Data, MLPB& Value) {
    Value = MLPB(BSerializer::Deserialize<size_t>(Data));
    for (size_t i = 0; i < Value.layerCount; ++i) {
        Value.SetLayer(i, MLPBL::Deserialize(Data));
    }
}