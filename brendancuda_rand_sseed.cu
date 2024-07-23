#include "brendancuda_rand_sseed.h"

__device__ uint32_t BrendanCUDA::Random::GetSeedOnKernel(uint32_t BaseSeed) {
    constexpr uint32_t m0 = 1172153037;
    constexpr uint32_t m1 = 1306592842;
    constexpr uint32_t m2 = 1313710305;
    constexpr uint32_t m3 = 3637357845;
    constexpr uint32_t m4 = 2676614147;
    constexpr uint32_t m5 = 3447833080;
    constexpr uint32_t m6 = 218029459;

    constexpr uint32_t s0 = 21;
    constexpr uint32_t s1 = 17;
    constexpr uint32_t s2 = 6;
    constexpr uint32_t s4 = 27;
    constexpr uint32_t s5 = 20;
    constexpr uint32_t s6 = 15;

    constexpr uint32_t x0 = 427380630;
    constexpr uint32_t x1 = 2022110166;
    constexpr uint32_t x2 = 1054627545;
    constexpr uint32_t x3 = 898584167;
    constexpr uint32_t x4 = 857594692;
    constexpr uint32_t x5 = 2625149580;
    constexpr uint32_t x6 = 3099295887;

    uint32_t v0 = (BaseSeed ^ x0) * m0;
    uint32_t v1 = (threadIdx.x ^ x1) * m1;
    uint32_t v2 = (threadIdx.y ^ x2) * m2;
    uint32_t v3 = (threadIdx.z ^ x3) * m3;
    uint32_t v4 = (blockIdx.x ^ x4) * m4;
    uint32_t v5 = (blockIdx.y ^ x5) * m5;
    uint32_t v6 = (blockIdx.z ^ x6) * m6;

    return
        ((v0 << s0) | (v0 >> (32 - s0))) ^
        ((v1 << s1) | (v1 >> (32 - s1))) ^
        ((v2 << s2) | (v2 >> (32 - s2))) ^
        v3 ^
        ((v4 << s4) | (v4 >> (32 - s4))) ^
        ((v5 << s5) | (v5 >> (32 - s5))) ^
        ((v6 << s6) | (v6 >> (32 - s6)));
}
__device__ uint64_t BrendanCUDA::Random::GetSeedOnKernel(uint64_t BaseSeed) {
    constexpr uint64_t m0 = 4577840436625149039ui64;
    constexpr uint64_t m1 = 6272590855536809777ui64;
    constexpr uint64_t m2 = 3819577067935900843ui64;
    constexpr uint64_t m3 = 6495922857260069574ui64;
    constexpr uint64_t m4 = 15911984094782298837ui64;
    constexpr uint64_t m5 = 1199187422582148453ui64;
    constexpr uint64_t m6 = 9056037786491530174ui64;

    constexpr uint32_t s0 = 3;
    constexpr uint32_t s1 = 7;
    constexpr uint32_t s2 = 5;
    constexpr uint32_t s3 = 39;
    constexpr uint32_t s4 = 28;
    constexpr uint32_t s5 = 54;

    constexpr uint64_t x0 = 12094415567977192229ui64;
    constexpr uint64_t x1 = 8485418671648197580ui64;
    constexpr uint64_t x2 = 5236607620774634909ui64;
    constexpr uint64_t x3 = 12231434909571960088ui64;
    constexpr uint64_t x4 = 17195966273802230455ui64;
    constexpr uint64_t x5 = 18272042865494684768ui64;
    constexpr uint64_t x6 = 818843249305457522ui64;

    uint64_t v0 = (BaseSeed ^ x0) * m0;
    uint64_t v1 = (threadIdx.x ^ x1) * m1;
    uint64_t v2 = (threadIdx.y ^ x2) * m2;
    uint64_t v3 = (threadIdx.z ^ x3) * m3;
    uint64_t v4 = (blockIdx.x ^ x4) * m4;
    uint64_t v5 = (blockIdx.y ^ x5) * m5;
    uint64_t v6 = (blockIdx.z ^ x6) * m6;

    return
        ((v0 << s0) | (v0 >> (64 - s0))) ^
        ((v1 << s1) | (v1 >> (64 - s1))) ^
        ((v2 << s2) | (v2 >> (64 - s2))) ^
        ((v3 << s3) | (v3 >> (64 - s3))) ^
        ((v4 << s4) | (v4 >> (64 - s4))) ^
        ((v5 << s5) | (v5 >> (64 - s5))) ^
        v6;
}
__host__ __device__ uint64_t BrendanCUDA::Random::HashI64(uint64_t Value) {
    constexpr uint64_t m0 = 17602768720006943520ui64;
    constexpr uint64_t m1 = 12661310132110239234ui64;
    constexpr uint64_t m2 = 15558628490694610577ui64;

    constexpr uint32_t s0 = 24;
    constexpr uint32_t s1 = 6;
    constexpr uint32_t s2 = 7;

    uint64_t p0 = Value * m0;
    uint64_t p1 = Value * m1;
    uint64_t p2 = Value * m2;

    return
        ((p0 << s0) | (p0 >> (32 - s0))) ^
        ((p1 << s1) | (p1 >> (32 - s1))) ^
        ((p2 << s2) | (p2 >> (32 - s2)));
}