#include "brendancuda_random_bits.cuh"

__host__ __device__ uint64_t BrendanCUDA::Random::Get64Bits(uint32_t ProbabilityOf1, rngWState<uint64_t> rng) {
    uint32_t ct = BrendanCUDA::Binary::CountBitsB(ProbabilityOf1);
    if (!ct) {
        return 0;
    }
    uint32_t lb = 1u << 31 >> (ct - 1);
    uint64_t cr = rng.Run();
    for (uint32_t i = 1ui64 << 31; i > lb; i >>= 1) {
        if (ProbabilityOf1 & i) {
            cr |= rng.Run();
        }
        else {
            cr &= rng.Run();
        }
    }
    return cr;
}