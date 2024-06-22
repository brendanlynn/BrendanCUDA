#include "brendancuda_ai_mlpb_mlpbl.cuh"

__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL() {
    type = networkType_t::eNull;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL8T8 MLPBL) {
    type = eMLPBL8T8;
    d8t8 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL8T16 MLPBL) {
    type = eMLPBL8T16;
    d8t16 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL8T32 MLPBL) {
    type = eMLPBL8T32;
    d8t32 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL8T64 MLPBL) {
    type = eMLPBL8T64;
    d8t64 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL16T8 MLPBL) {
    type = eMLPBL16T8;
    d16t8 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL16T16 MLPBL) {
    type = eMLPBL16T16;
    d16t16 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL16T32 MLPBL) {
    type = eMLPBL16T32;
    d16t32 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL16T64 MLPBL) {
    type = eMLPBL16T64;
    d16t64 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL32T8 MLPBL) {
    type = eMLPBL32T8;
    d32t8 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL32T16 MLPBL) {
    type = eMLPBL32T16;
    d32t16 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL32T32 MLPBL) {
    type = eMLPBL32T32;
    d32t32 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL32T64 MLPBL) {
    type = eMLPBL32T64;
    d32t64 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL64T8 MLPBL) {
    type = eMLPBL64T8;
    d64t8 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL64T16 MLPBL) {
    type = eMLPBL64T16;
    d64t16 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL64T32 MLPBL) {
    type = eMLPBL64T32;
    d64t32 = MLPBL;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL::MLPBL(MLPBL64T64 MLPBL) {
    type = eMLPBL64T64;
    d64t64 = MLPBL;
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL::Run(uint64_t Input) const {
    switch (type) {
    case eMLPBL8T8:
        return d8t8.RunG(Input);
    case eMLPBL8T16:
        return d8t16.RunG(Input);
    case eMLPBL8T32:
        return d8t32.RunG(Input);
    case eMLPBL8T64:
        return d8t64.RunG(Input);
    case eMLPBL16T8:
        return d16t8.RunG(Input);
    case eMLPBL16T16:
        return d16t16.RunG(Input);
    case eMLPBL16T32:
        return d16t32.RunG(Input);
    case eMLPBL16T64:
        return d16t64.RunG(Input);
    case eMLPBL32T8:
        return d32t8.RunG(Input);
    case eMLPBL32T16:
        return d32t16.RunG(Input);
    case eMLPBL32T32:
        return d32t32.RunG(Input);
    case eMLPBL32T64:
        return d32t64.RunG(Input);
    case eMLPBL64T8:
        return d64t8.RunG(Input);
    case eMLPBL64T16:
        return d64t16.RunG(Input);
    case eMLPBL64T32:
        return d64t32.RunG(Input);
    case eMLPBL64T64:
        return d64t64.RunG(Input);
    default:
        return 0l;
    }
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL::Dispose() {
    switch (type) {
    case eMLPBL8T8:
        return d8t8.Dispose();
    case eMLPBL8T16:
        return d8t16.Dispose();
    case eMLPBL8T32:
        return d8t32.Dispose();
    case eMLPBL8T64:
        return d8t64.Dispose();
    case eMLPBL16T8:
        return d16t8.Dispose();
    case eMLPBL16T16:
        return d16t16.Dispose();
    case eMLPBL16T32:
        return d16t32.Dispose();
    case eMLPBL16T64:
        return d16t64.Dispose();
    case eMLPBL32T8:
        return d32t8.Dispose();
    case eMLPBL32T16:
        return d32t16.Dispose();
    case eMLPBL32T32:
        return d32t32.Dispose();
    case eMLPBL32T64:
        return d32t64.Dispose();
    case eMLPBL64T8:
        return d64t8.Dispose();
    case eMLPBL64T16:
        return d64t16.Dispose();
    case eMLPBL64T32:
        return d64t32.Dispose();
    case eMLPBL64T64:
        return d64t64.Dispose();
    default:
        return;
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL BrendanCUDA::AI::MLPB::MLPBL::Clone() {
    switch (type) {
    case eMLPBL8T8:
        return MLPBL(d8t8.Clone());
    case eMLPBL8T16:
        return MLPBL(d8t16.Clone());
    case eMLPBL8T32:
        return MLPBL(d8t32.Clone());
    case eMLPBL8T64:
        return MLPBL(d8t64.Clone());
    case eMLPBL16T8:
        return MLPBL(d16t8.Clone());
    case eMLPBL16T16:
        return MLPBL(d16t16.Clone());
    case eMLPBL16T32:
        return MLPBL(d16t32.Clone());
    case eMLPBL16T64:
        return MLPBL(d16t64.Clone());
    case eMLPBL32T8:
        return MLPBL(d32t8.Clone());
    case eMLPBL32T16:
        return MLPBL(d32t16.Clone());
    case eMLPBL32T32:
        return MLPBL(d32t32.Clone());
    case eMLPBL32T64:
        return MLPBL(d32t64.Clone());
    case eMLPBL64T8:
        return MLPBL(d64t8.Clone());
    case eMLPBL64T16:
        return MLPBL(d64t16.Clone());
    case eMLPBL64T32:
        return MLPBL(d64t32.Clone());
    case eMLPBL64T64:
        return MLPBL(d64t64.Clone());
    default:
        return MLPBL();
    }
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
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
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL BrendanCUDA::AI::MLPB::MLPBL::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    switch (type) {
    case eMLPBL8T8:
        return MLPBL(d8t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T16:
        return MLPBL(d8t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T32:
        return MLPBL(d8t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T64:
        return MLPBL(d8t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T8:
        return MLPBL(d16t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T16:
        return MLPBL(d16t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T32:
        return MLPBL(d16t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T64:
        return MLPBL(d16t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T8:
        return MLPBL(d32t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T16:
        return MLPBL(d32t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T32:
        return MLPBL(d32t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T64:
        return MLPBL(d32t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T8:
        return MLPBL(d64t8.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T16:
        return MLPBL(d64t16.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T32:
        return MLPBL(d64t32.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T64:
        return MLPBL(d64t64.ReproduceWFlips(WeightsFlipProb, BiasFlipProb, RNG));
    default:
        return MLPBL();
    }
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
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
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL BrendanCUDA::AI::MLPB::MLPBL::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    switch (type) {
    case eMLPBL8T8:
        return MLPBL(d8t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T16:
        return MLPBL(d8t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T32:
        return MLPBL(d8t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL8T64:
        return MLPBL(d8t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T8:
        return MLPBL(d16t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T16:
        return MLPBL(d16t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T32:
        return MLPBL(d16t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL16T64:
        return MLPBL(d16t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T8:
        return MLPBL(d32t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T16:
        return MLPBL(d32t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T32:
        return MLPBL(d32t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL32T64:
        return MLPBL(d32t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T8:
        return MLPBL(d64t8.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T16:
        return MLPBL(d64t16.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T32:
        return MLPBL(d64t32.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    case eMLPBL64T64:
        return MLPBL(d64t64.ReproduceWTargets(WeightsEachFlipProb, BiasFlipProb, RNG));
    default:
        return MLPBL();
    }
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) {
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
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL BrendanCUDA::AI::MLPB::MLPBL::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const {
    switch (type) {
    case eMLPBL8T8:
        return MLPBL(d8t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T16:
        return MLPBL(d8t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T32:
        return MLPBL(d8t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL8T64:
        return MLPBL(d8t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T8:
        return MLPBL(d16t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T16:
        return MLPBL(d16t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T32:
        return MLPBL(d16t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL16T64:
        return MLPBL(d16t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T8:
        return MLPBL(d32t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T16:
        return MLPBL(d32t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T32:
        return MLPBL(d32t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL32T64:
        return MLPBL(d32t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T8:
        return MLPBL(d64t8.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T16:
        return MLPBL(d64t16.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T32:
        return MLPBL(d64t32.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    case eMLPBL64T64:
        return MLPBL(d64t64.ReproduceWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG));
    default:
        return MLPBL();
    }
}