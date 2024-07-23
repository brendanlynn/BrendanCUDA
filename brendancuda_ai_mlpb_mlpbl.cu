#include "brendancuda_ai_mlpb_mlpbl.h"

__host__ __device__ __forceinline uint64_t applyTargetFlipsTo1s_getEdits(uint64_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
    if (rn1 < FlipProb) {
        uint32_t mrn = rn2 % CountOf1s;

        for (uint64_t m = 1ui64; m; m <<= 1) {
            if (m & Value) {
                --mrn;
                if (!mrn) {
                    return m;
                }
            }
        }
    }
    return 0;
}
__host__ __device__ __forceinline uint32_t applyTargetFlipsTo1s_getEdits(uint32_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
    if (rn1 < FlipProb) {
        uint32_t mrn = rn2 % CountOf1s;

        for (uint32_t m = 1ui32; m; m <<= 1) {
            if (m & Value) {
                --mrn;
                if (!mrn) {
                    return m;
                }
            }
        }
    }
    return 0;
}
__host__ __device__ __forceinline uint16_t applyTargetFlipsTo1s_getEdits(uint16_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
    if (rn1 < FlipProb) {
        uint32_t mrn = rn2 % CountOf1s;

        for (uint16_t m = 1ui16; m; m <<= 1) {
            if (m & Value) {
                --mrn;
                if (!mrn) {
                    return m;
                }
            }
        }
    }
    return 0;
}
__host__ __device__ __forceinline uint8_t applyTargetFlipsTo1s_getEdits(uint8_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
    if (rn1 < FlipProb) {
        uint32_t mrn = rn2 % CountOf1s;

        for (uint8_t m = 1ui8; m; m <<= 1) {
            if (m & Value) {
                --mrn;
                if (!mrn) {
                    return m;
                }
            }
        }
    }
    return 0;
}

__host__ __device__ __forceinline uint64_t applyTargetFlips(uint64_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
    if (Value == 0) {
        if (rn1 < FlipProb) {
            return 1ui64 << (rn2 & 63);
        }
        return 0;
    }
    else if (Value == 0xFFFFFFFFFFFFFFFFui64) {
        if (rn3 < FlipProb) {
            return ~(1ui64 << (rn4 & 63));
        }
        return 0xFFFFFFFFFFFFFFFFui64;
    }
    else {
        uint32_t v1c = std::popcount(Value);

        return
            Value ^
            applyTargetFlipsTo1s_getEdits(Value, v1c, FlipProb, rn1, rn2) ^
            applyTargetFlipsTo1s_getEdits(~Value, 64 - v1c, FlipProb, rn3, rn4);
    }
}
__host__ __device__ __forceinline uint32_t applyTargetFlips(uint32_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
    if (Value == 0) {
        if (rn1 < FlipProb) {
            return 1ui32 << (rn2 & 31);
        }
        return 0;
    }
    else if (Value == 0xFFFFFFFFui32) {
        if (rn3 < FlipProb) {
            return ~(1ui32 << (rn4 & 31));
        }
        return 0xFFFFFFFFui32;
    }
    else {
        uint32_t v1c = std::popcount(Value);

        return
            Value ^
            applyTargetFlipsTo1s_getEdits(Value, v1c, FlipProb, rn1, rn2) ^
            applyTargetFlipsTo1s_getEdits(~Value, 32 - v1c, FlipProb, rn3, rn4);
    }
}
__host__ __device__ __forceinline uint16_t applyTargetFlips(uint16_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
    if (Value == 0) {
        if (rn1 < FlipProb) {
            return 1ui16 << (rn2 & 15);
        }
        return 0;
    }
    else if (Value == 0xFFFFui16) {
        if (rn3 < FlipProb) {
            return ~(1ui16 << (rn4 & 15));
        }
        return 0xFFFFui16;
    }
    else {
        uint32_t v1c = std::popcount(Value);

        return
            Value ^
            applyTargetFlipsTo1s_getEdits(Value, v1c, FlipProb, rn1, rn2) ^
            applyTargetFlipsTo1s_getEdits((uint16_t)~Value, 16 - v1c, FlipProb, rn3, rn4);
    }
}
__host__ __device__ __forceinline uint8_t applyTargetFlips(uint8_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
    if (Value == 0) {
        if (rn1 < FlipProb) {
            return 1ui8 << (rn2 & 7);
        }
        return 0;
    }
    else if (Value == 0xFFui8) {
        if (rn3 < FlipProb) {
            return ~(1ui8 << (rn4 & 7));
        }
        return 0xFFui8;
    }
    else {
        uint32_t v1c = std::popcount(Value);

        return
            Value ^
            applyTargetFlipsTo1s_getEdits(Value, v1c, FlipProb, rn1, rn2) ^
            applyTargetFlipsTo1s_getEdits((uint8_t)~Value, 8 - v1c, FlipProb, rn3, rn4);
    }
}

__global__ void applyTargetFlipsOnArray_kernel(uint64_t* arr, uint32_t flipProb, uint64_t bs) {
    uint64_t& v(arr[blockIdx.x]);

    uint64_t rn64_1 = BrendanCUDA::Random::HashI64(BrendanCUDA::Random::GetSeedOnKernel(bs));
    uint64_t rn64_2 = BrendanCUDA::Random::HashI64(rn64_1 ^ 12210506935820558677);

    uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
    uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
    uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
    uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

    v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
}
__global__ void applyTargetFlipsOnArray_kernel(uint32_t* arr, uint32_t flipProb, uint64_t bs) {
    uint32_t& v(arr[blockIdx.x]);

    uint64_t rn64_1 = BrendanCUDA::Random::HashI64(BrendanCUDA::Random::GetSeedOnKernel(bs));
    uint64_t rn64_2 = BrendanCUDA::Random::HashI64(rn64_1 ^ 484654973014905267);

    uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
    uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
    uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
    uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

    v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
}
__global__ void applyTargetFlipsOnArray_kernel(uint16_t* arr, uint32_t flipProb, uint64_t bs) {
    uint16_t& v(arr[blockIdx.x]);

    uint64_t rn64_1 = BrendanCUDA::Random::HashI64(BrendanCUDA::Random::GetSeedOnKernel(bs));
    uint64_t rn64_2 = BrendanCUDA::Random::HashI64(rn64_1 ^ 3123193471197220784);

    uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
    uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
    uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
    uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

    v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
}
__global__ void applyTargetFlipsOnArray_kernel(uint8_t* arr, uint32_t flipProb, uint64_t bs) {
    uint8_t& v(arr[blockIdx.x]);

    uint64_t rn64_1 = BrendanCUDA::Random::HashI64(BrendanCUDA::Random::GetSeedOnKernel(bs));
    uint64_t rn64_2 = BrendanCUDA::Random::HashI64(rn64_1 ^ 11199430323554825400);

    uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
    uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
    uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
    uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

    v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
}

__host__ __device__ __forceinline void BrendanCUDA::details::applyTargetFlipsOnArray(uint64_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG) {
#ifdef __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint64_t& v(arr[i]);

        uint64_t rn64_1 = RNG();
        uint64_t rn64_2 = RNG();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel << <sz, 1 >> > (arr, flipProb, RNG());
#endif
}
__host__ __device__ __forceinline void BrendanCUDA::details::applyTargetFlipsOnArray(uint32_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG) {
#ifdef __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint32_t& v(arr[i]);

        uint64_t rn64_1 = RNG();
        uint64_t rn64_2 = RNG();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel << <sz, 1 >> > (arr, flipProb, RNG());
#endif
}
__host__ __device__ __forceinline void BrendanCUDA::details::applyTargetFlipsOnArray(uint16_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG) {
#ifdef __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint16_t& v(arr[i]);

        uint64_t rn64_1 = RNG();
        uint64_t rn64_2 = RNG();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel << <sz, 1 >> > (arr, flipProb, RNG());
#endif
}
__host__ __device__ __forceinline void BrendanCUDA::details::applyTargetFlipsOnArray(uint8_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG) {
#ifdef __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint8_t& v(arr[i]);

        uint64_t rn64_1 = RNG();
        uint64_t rn64_2 = RNG();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel << <sz, 1 >> > (arr, flipProb, RNG());
#endif
}