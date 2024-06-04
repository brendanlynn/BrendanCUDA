#pragma once

#include <memory>
#include "brendancuda_ai_mlpb_mlpbl.cuh"
#include "brendancuda_random_rngfunc.cuh"
#include "brendancuda_macros.cuh"

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
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPB ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPB ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPB ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ static MLPBL Deserialize(std::basic_istream<char>& Stream);
            private:
                MLPBL* layers;
                size_t layerCount;
            };
        }
    }
}