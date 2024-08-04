#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "brendancuda_ai_mlpb_mlpbl.h"
#include "brendancuda_rand_anyrng.h"

namespace BrendanCUDA {
    namespace AI {
        namespace MLPB {
            class alignas(8) MLPBLW final {
            public:
                __host__ __device__ __forceinline MLPBLW();
                template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
                __host__ __device__ __forceinline MLPBLW(MLPBL<_TInput, _TOutput> MLPBL);

                __host__ __device__ uint64_t Run(uint64_t Input) const;

                __host__ __device__ void Dispose();

                __host__ __device__ MLPBLW Clone() const;
                __host__ __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBLW ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBLW ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPBLW ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                size_t SerializedSize() const;
                void Serialize(void*& Data) const;
                static MLPBLW Deserialize(const void*& Data);
                static __forceinline void Deserialize(const void*& Data, void* ObjMem);

                union {
                    mlpbl8T8_t d8t8;
                    mlpbl8T16_t d8t16;
                    mlpbl8T32_t d8t32;
                    mlpbl8T64_t d8t64;
                    mlpbl16T8_t d16t8;
                    mlpbl16T16_t d16t16;
                    mlpbl16T32_t d16t32;
                    mlpbl16T64_t d16t64;
                    mlpbl32T8_t d32t8;
                    mlpbl32T16_t d32t16;
                    mlpbl32T32_t d32t32;
                    mlpbl32T64_t d32t64;
                    mlpbl64T8_t d64t8;
                    mlpbl64T16_t d64t16;
                    mlpbl64T32_t d64t32;
                    mlpbl64T64_t d64t64;
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

__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBLW::MLPBLW() {
    type = networkType_t::eNull;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBLW::MLPBLW(MLPBL<_TInput, _TOutput> MLPBL) {
    if constexpr (std::same_as<_TInput, uint8_t>) {
        if constexpr (std::same_as<_TOutput, uint8_t>) {
            type = eMLPBL8T8;
            d8t8 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint16_t>) {
            type = eMLPBL8T16;
            d8t16 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint32_t>) {
            type = eMLPBL8T32;
            d8t32 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint64_t>) {
            type = eMLPBL8T64;
            d8t64 = MLPBL;
        }
    }
    else if constexpr (std::same_as<_TInput, uint16_t>) {
        if constexpr (std::same_as<_TOutput, uint8_t>) {
            type = eMLPBL16T8;
            d16t8 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint16_t>) {
            type = eMLPBL16T16;
            d16t16 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint32_t>) {
            type = eMLPBL16T32;
            d16t32 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint64_t>) {
            type = eMLPBL16T64;
            d16t64 = MLPBL;
        }
    }
    else if constexpr (std::same_as<_TInput, uint32_t>) {
        if constexpr (std::same_as<_TOutput, uint8_t>) {
            type = eMLPBL32T8;
            d32t8 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint16_t>) {
            type = eMLPBL32T16;
            d32t16 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint32_t>) {
            type = eMLPBL32T32;
            d32t32 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint64_t>) {
            type = eMLPBL32T64;
            d32t64 = MLPBL;
        }
    }
    else if constexpr (std::same_as<_TInput, uint64_t>) {
        if constexpr (std::same_as<_TOutput, uint8_t>) {
            type = eMLPBL64T8;
            d64t8 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint16_t>) {
            type = eMLPBL64T16;
            d64t16 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint32_t>) {
            type = eMLPBL64T32;
            d64t32 = MLPBL;
        }
        else if constexpr (std::same_as<_TOutput, uint64_t>) {
            type = eMLPBL64T64;
            d64t64 = MLPBL;
        }
    }
}
__forceinline void BrendanCUDA::AI::MLPB::MLPBLW::Deserialize(const void*& Data, void* ObjMem) {
    new (ObjMem) MLPBLW(Deserialize(Data));
}