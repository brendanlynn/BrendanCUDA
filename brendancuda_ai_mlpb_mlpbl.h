#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "brendancuda_ai_mlpb_mlpbls.h"
#include "brendancuda_rand_anyrng.h"

namespace BrendanCUDA {
    namespace AI {
        namespace MLPB {
            class alignas(8) MLPBL final {
            public:
                __host__ __device__ MLPBL();
                __host__ __device__ MLPBL(MLPBL8T8 MLPBL);
                __host__ __device__ MLPBL(MLPBL8T16 MLPBL);
                __host__ __device__ MLPBL(MLPBL8T32 MLPBL);
                __host__ __device__ MLPBL(MLPBL8T64 MLPBL);
                __host__ __device__ MLPBL(MLPBL16T8 MLPBL);
                __host__ __device__ MLPBL(MLPBL16T16 MLPBL);
                __host__ __device__ MLPBL(MLPBL16T32 MLPBL);
                __host__ __device__ MLPBL(MLPBL16T64 MLPBL);
                __host__ __device__ MLPBL(MLPBL32T8 MLPBL);
                __host__ __device__ MLPBL(MLPBL32T16 MLPBL);
                __host__ __device__ MLPBL(MLPBL32T32 MLPBL);
                __host__ __device__ MLPBL(MLPBL32T64 MLPBL);
                __host__ __device__ MLPBL(MLPBL64T8 MLPBL);
                __host__ __device__ MLPBL(MLPBL64T16 MLPBL);
                __host__ __device__ MLPBL(MLPBL64T32 MLPBL);
                __host__ __device__ MLPBL(MLPBL64T64 MLPBL);

                __host__ __device__ uint64_t Run(uint64_t Input) const;

                __host__ __device__ void Dispose();

                __host__ __device__ MLPBL Clone();
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBL ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBL Deserialize(const void*& Data);

                union {
                    MLPBL8T8 d8t8;
                    MLPBL8T16 d8t16;
                    MLPBL8T32 d8t32;
                    MLPBL8T64 d8t64;
                    MLPBL16T8 d16t8;
                    MLPBL16T16 d16t16;
                    MLPBL16T32 d16t32;
                    MLPBL16T64 d16t64;
                    MLPBL32T8 d32t8;
                    MLPBL32T16 d32t16;
                    MLPBL32T32 d32t32;
                    MLPBL32T64 d32t64;
                    MLPBL64T8 d64t8;
                    MLPBL64T16 d64t16;
                    MLPBL64T32 d64t32;
                    MLPBL64T64 d64t64;
                };
                enum networkType_t : int32_t {
                    eNull = -1,
                    eMLPBL8T8,
                    eMLPBL8T16,
                    eMLPBL8T32,
                    eMLPBL8T64,
                    eMLPBL16T8,
                    eMLPBL16T16,
                    eMLPBL16T32,
                    eMLPBL16T64,
                    eMLPBL32T8,
                    eMLPBL32T16,
                    eMLPBL32T32,
                    eMLPBL32T64,
                    eMLPBL64T8,
                    eMLPBL64T16,
                    eMLPBL64T32,
                    eMLPBL64T64,
                };
                networkType_t type;
            };
        }
    }
}