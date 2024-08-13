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
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                __device__ void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, curandState& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ MLPBLW ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const;
#ifdef __CUDACC__
                __device__ MLPBLW ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, curandState& RNG) const;
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
#ifdef __CUDACC__
                __device__ void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, curandState& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ MLPBLW ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const;
#ifdef __CUDACC__
                __device__ MLPBLW ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, curandState& RNG) const;
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
#ifdef __CUDACC__
                __device__ void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, curandState& RNG);
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ MLPBLW ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) const;
#ifdef __CUDACC__
                __device__ MLPBLW ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, curandState& RNG) const;
#endif

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
template <std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::AI::MLPB::MLPBLW::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    switch (type) {
    case eMLPBL8T8:
        d8t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T16:
        d8t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T32:
        d8t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T64:
        d8t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T8:
        d16t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T16:
        d16t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T32:
        d16t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T64:
        d16t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T8:
        d32t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T16:
        d32t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T32:
        d32t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T64:
        d32t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T8:
        d64t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T16:
        d64t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T32:
        d64t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T64:
        d64t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::AI::MLPB::MLPBLW::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, curandState& RNG) {
    switch (type) {
    case eMLPBL8T8:
        d8t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T16:
        d8t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T32:
        d8t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T64:
        d8t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T8:
        d16t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T16:
        d16t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T32:
        d16t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T64:
        d16t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T8:
        d32t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T16:
        d32t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T32:
        d32t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T64:
        d32t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T8:
        d64t8.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T16:
        d64t16.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T32:
        d64t32.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T64:
        d64t64.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
        break;
    }
}
#endif
template <std::uniform_random_bit_generator _TRNG>
__host__ auto BrendanCUDA::AI::MLPB::MLPBLW::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const -> MLPBLW {
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(d8t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T16:
        return MLPBLW(d8t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T32:
        return MLPBLW(d8t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T64:
        return MLPBLW(d8t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T8:
        return MLPBLW(d16t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T16:
        return MLPBLW(d16t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T32:
        return MLPBLW(d16t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T64:
        return MLPBLW(d16t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T8:
        return MLPBLW(d32t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T16:
        return MLPBLW(d32t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T32:
        return MLPBLW(d32t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T64:
        return MLPBLW(d32t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T8:
        return MLPBLW(d64t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T16:
        return MLPBLW(d64t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T32:
        return MLPBLW(d64t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T64:
        return MLPBLW(d64t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    default:
        return MLPBLW();
    }
}
#ifdef __CUDACC__
__device__ auto BrendanCUDA::AI::MLPB::MLPBLW::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, curandState& RNG) const -> MLPBLW {
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(d8t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T16:
        return MLPBLW(d8t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T32:
        return MLPBLW(d8t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T64:
        return MLPBLW(d8t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T8:
        return MLPBLW(d16t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T16:
        return MLPBLW(d16t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T32:
        return MLPBLW(d16t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T64:
        return MLPBLW(d16t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T8:
        return MLPBLW(d32t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T16:
        return MLPBLW(d32t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T32:
        return MLPBLW(d32t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T64:
        return MLPBLW(d32t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T8:
        return MLPBLW(d64t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T16:
        return MLPBLW(d64t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T32:
        return MLPBLW(d64t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T64:
        return MLPBLW(d64t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    default:
        return MLPBLW();
    }
}
#endif
template <std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::AI::MLPB::MLPBLW::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    switch (type) {
    case eMLPBL8T8:
        d8t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T16:
        d8t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T32:
        d8t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T64:
        d8t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T8:
        d16t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T16:
        d16t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T32:
        d16t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T64:
        d16t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T8:
        d32t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T16:
        d32t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T32:
        d32t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T64:
        d32t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T8:
        d64t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T16:
        d64t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T32:
        d64t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T64:
        d64t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::AI::MLPB::MLPBLW::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, curandState& RNG) {
    switch (type) {
    case eMLPBL8T8:
        d8t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T16:
        d8t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T32:
        d8t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL8T64:
        d8t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T8:
        d16t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T16:
        d16t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T32:
        d16t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL16T64:
        d16t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T8:
        d32t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T16:
        d32t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T32:
        d32t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL32T64:
        d32t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T8:
        d64t8.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T16:
        d64t16.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T32:
        d64t32.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    case eMLPBL64T64:
        d64t64.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
        break;
    }
}
#endif
template <std::uniform_random_bit_generator _TRNG>
__host__ auto BrendanCUDA::AI::MLPB::MLPBLW::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const -> MLPBLW {
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(d8t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T16:
        return MLPBLW(d8t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T32:
        return MLPBLW(d8t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T64:
        return MLPBLW(d8t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T8:
        return MLPBLW(d16t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T16:
        return MLPBLW(d16t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T32:
        return MLPBLW(d16t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T64:
        return MLPBLW(d16t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T8:
        return MLPBLW(d32t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T16:
        return MLPBLW(d32t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T32:
        return MLPBLW(d32t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T64:
        return MLPBLW(d32t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T8:
        return MLPBLW(d64t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T16:
        return MLPBLW(d64t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T32:
        return MLPBLW(d64t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T64:
        return MLPBLW(d64t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    default:
        return MLPBLW();
    }
}
#ifdef __CUDACC__
__device__ auto BrendanCUDA::AI::MLPB::MLPBLW::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, curandState& RNG) const -> MLPBLW {
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(d8t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T16:
        return MLPBLW(d8t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T32:
        return MLPBLW(d8t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T64:
        return MLPBLW(d8t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T8:
        return MLPBLW(d16t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T16:
        return MLPBLW(d16t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T32:
        return MLPBLW(d16t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T64:
        return MLPBLW(d16t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T8:
        return MLPBLW(d32t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T16:
        return MLPBLW(d32t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T32:
        return MLPBLW(d32t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T64:
        return MLPBLW(d32t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T8:
        return MLPBLW(d64t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T16:
        return MLPBLW(d64t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T32:
        return MLPBLW(d64t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T64:
        return MLPBLW(d64t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    default:
        return MLPBLW();
    }
}
#endif
template <std::uniform_random_bit_generator _TRNG>
__host__ void BrendanCUDA::AI::MLPB::MLPBLW::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    switch (type) {
    case eMLPBL8T8:
        d8t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL8T16:
        d8t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL8T32:
        d8t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL8T64:
        d8t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T8:
        d16t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T16:
        d16t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T32:
        d16t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T64:
        d16t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T8:
        d32t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T16:
        d32t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T32:
        d32t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T64:
        d32t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T8:
        d64t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T16:
        d64t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T32:
        d64t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T64:
        d64t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    }
}
#ifdef __CUDACC__
__device__ void BrendanCUDA::AI::MLPB::MLPBLW::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, curandState& RNG) {
    switch (type) {
    case eMLPBL8T8:
        d8t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL8T16:
        d8t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL8T32:
        d8t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL8T64:
        d8t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T8:
        d16t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T16:
        d16t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T32:
        d16t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL16T64:
        d16t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T8:
        d32t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T16:
        d32t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T32:
        d32t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL32T64:
        d32t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T8:
        d64t8.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T16:
        d64t16.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T32:
        d64t32.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    case eMLPBL64T64:
        d64t64.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
        break;
    }
}
#endif
template <std::uniform_random_bit_generator _TRNG>
__host__ auto BrendanCUDA::AI::MLPB::MLPBLW::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) const -> MLPBLW {
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(d8t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T16:
        return MLPBLW(d8t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T32:
        return MLPBLW(d8t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T64:
        return MLPBLW(d8t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T8:
        return MLPBLW(d16t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T16:
        return MLPBLW(d16t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T32:
        return MLPBLW(d16t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T64:
        return MLPBLW(d16t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T8:
        return MLPBLW(d32t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T16:
        return MLPBLW(d32t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T32:
        return MLPBLW(d32t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T64:
        return MLPBLW(d32t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T8:
        return MLPBLW(d64t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T16:
        return MLPBLW(d64t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T32:
        return MLPBLW(d64t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T64:
        return MLPBLW(d64t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    default:
        return MLPBLW();
    }
}
#ifdef __CUDACC__
__device__ auto BrendanCUDA::AI::MLPB::MLPBLW::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, curandState& RNG) const -> MLPBLW {
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(d8t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T16:
        return MLPBLW(d8t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T32:
        return MLPBLW(d8t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T64:
        return MLPBLW(d8t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T8:
        return MLPBLW(d16t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T16:
        return MLPBLW(d16t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T32:
        return MLPBLW(d16t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T64:
        return MLPBLW(d16t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T8:
        return MLPBLW(d32t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T16:
        return MLPBLW(d32t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T32:
        return MLPBLW(d32t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T64:
        return MLPBLW(d32t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T8:
        return MLPBLW(d64t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T16:
        return MLPBLW(d64t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T32:
        return MLPBLW(d64t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T64:
        return MLPBLW(d64t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    default:
        return MLPBLW();
    }
}
#endif
__forceinline void BrendanCUDA::AI::MLPB::MLPBLW::Deserialize(const void*& Data, void* ObjMem) {
    new (ObjMem) MLPBLW(Deserialize(Data));
}