#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <bit>
#include <concepts>
#include "BSerializer/Serializer.h"
#include "brendancuda_rand_anyrng.h"
#include "brendancuda_ai.h"
#include "brendancuda_binary_basic.h"
#include "brendancuda_rand_bits.h"
#include "brendancuda_errorhelp.h"
#include "brendancuda_dcopy.cuh"
#include "brendancuda_rand_sseed.h"

namespace BrendanCUDA {
    namespace details {
        __host__ __device__ __forceinline uint64_t applyTargetFlipsTo1s_getEdits(uint64_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2);
        __host__ __device__ __forceinline uint32_t applyTargetFlipsTo1s_getEdits(uint32_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2);
        __host__ __device__ __forceinline uint16_t applyTargetFlipsTo1s_getEdits(uint16_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2);
        __host__ __device__ __forceinline uint8_t applyTargetFlipsTo1s_getEdits(uint8_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2);

        __host__ __device__ __forceinline uint64_t applyTargetFlips(uint64_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4);
        __host__ __device__ __forceinline uint32_t applyTargetFlips(uint32_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4);
        __host__ __device__ __forceinline uint16_t applyTargetFlips(uint16_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4);
        __host__ __device__ __forceinline uint8_t applyTargetFlips(uint8_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4);

        __global__ void applyTargetFlipsOnArray_kernel(uint64_t* arr, uint32_t flipProb, uint64_t bs);
        __global__ void applyTargetFlipsOnArray_kernel(uint32_t* arr, uint32_t flipProb, uint64_t bs);
        __global__ void applyTargetFlipsOnArray_kernel(uint16_t* arr, uint32_t flipProb, uint64_t bs);
        __global__ void applyTargetFlipsOnArray_kernel(uint8_t* arr, uint32_t flipProb, uint64_t bs);

        __host__ __device__ __forceinline void applyTargetFlipsOnArray(uint64_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
        __host__ __device__ __forceinline void applyTargetFlipsOnArray(uint32_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
        __host__ __device__ __forceinline void applyTargetFlipsOnArray(uint16_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
        __host__ __device__ __forceinline void applyTargetFlipsOnArray(uint8_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
    }
    namespace AI {
        namespace MLPB {
            template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
            class MLPBL final {
            public:
                __host__ __device__ __forceinline MLPBL();
                __host__ __device__ __forceinline MLPBL(_TInput* Weights, _TOutput* Bias);

                __device__ static __forceinline MLPBL<_TInput, _TOutput> CloneFrom(_TInput* Weights, _TOutput* Bias);
                template <bool _CopyFromHost>
                __host__ static __forceinline MLPBL<_TInput, _TOutput> CloneFrom(_TInput* Weights, _TOutput* Bias);

                __host__ __device__ __forceinline void Dispose();

                __host__ __device__ __forceinline _TInput* Weights() const;
                __host__ __device__ __forceinline _TInput* Weight(size_t Index) const;

                __host__ __device__ __forceinline _TOutput* Bias() const;

                template <bool _CopyToHost>
                __host__ __forceinline _TInput* CopyOutWeights() const;
                template <bool _CopyToHost>
                __host__ __forceinline void CopyOutWeights(_TInput* Weights) const;
                __device__ __forceinline _TInput* CopyOutWeights() const;
                __device__ __forceinline void CopyOutWeights(_TInput* Weights) const;
                template <bool _CopyFromHost>
                __host__ __forceinline void CopyInWeights(_TInput* Weights);
                __device__ __forceinline void CopyInWeights(_TInput* Weights);
                __host__ __device__ __forceinline _TInput GetWeight(size_t Index) const;
                __host__ __device__ __forceinline void SetWeight(size_t Index, _TInput Weight);

                __host__ __device__ __forceinline _TOutput GetBias() const;
                __host__ __device__ __forceinline void SetBias(_TOutput Bias);

                __host__ __device__ __forceinline _TOutput Run(_TInput Input) const;
                __host__ __device__ __forceinline uint64_t RunG(uint64_t Input) const;

                __host__ __device__ __forceinline void CopyTo(MLPBL<_TInput, _TOutput> Other) const;
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> Clone() const;

                __host__ __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ __forceinline void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ __forceinline MLPBL<_TInput, _TOutput> ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const;

                __forceinline size_t SerializedSize() const;
                __forceinline void Serialize(void*& Data) const;
                static __forceinline MLPBL<_TInput, _TOutput> Deserialize(const void*& Data);
                static __forceinline void Deserialize(const void*& Data, MLPBL<_TInput, _TOutput>& Value);
            private:
                _TInput* weights;
                _TOutput* bias;
            };

            using mlpbl8T8_t = MLPBL<uint8_t, uint8_t>;
            using mlpbl8T16_t = MLPBL<uint8_t, uint16_t>;
            using mlpbl8T32_t = MLPBL<uint8_t, uint32_t>;
            using mlpbl8T64_t = MLPBL<uint8_t, uint64_t>;
            using mlpbl16T8_t = MLPBL<uint16_t, uint8_t>;
            using mlpbl16T16_t = MLPBL<uint16_t, uint16_t>;
            using mlpbl16T32_t = MLPBL<uint16_t, uint32_t>;
            using mlpbl16T64_t = MLPBL<uint16_t, uint64_t>;
            using mlpbl32T8_t = MLPBL<uint32_t, uint8_t>;
            using mlpbl32T16_t = MLPBL<uint32_t, uint16_t>;
            using mlpbl32T32_t = MLPBL<uint32_t, uint32_t>;
            using mlpbl32T64_t = MLPBL<uint32_t, uint64_t>;
            using mlpbl64T8_t = MLPBL<uint64_t, uint8_t>;
            using mlpbl64T16_t = MLPBL<uint64_t, uint16_t>;
            using mlpbl64T32_t = MLPBL<uint64_t, uint32_t>;
            using mlpbl64T64_t = MLPBL<uint64_t, uint64_t>;
        }
    }
}

__host__ __device__ __forceinline uint64_t BrendanCUDA::details::applyTargetFlipsTo1s_getEdits(uint64_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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
__host__ __device__ __forceinline uint32_t BrendanCUDA::details::applyTargetFlipsTo1s_getEdits(uint32_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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
__host__ __device__ __forceinline uint16_t BrendanCUDA::details::applyTargetFlipsTo1s_getEdits(uint16_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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
__host__ __device__ __forceinline uint8_t BrendanCUDA::details::applyTargetFlipsTo1s_getEdits(uint8_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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

__host__ __device__ __forceinline uint64_t BrendanCUDA::details::applyTargetFlips(uint64_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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
__host__ __device__ __forceinline uint32_t BrendanCUDA::details::applyTargetFlips(uint32_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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
__host__ __device__ __forceinline uint16_t BrendanCUDA::details::applyTargetFlips(uint16_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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
__host__ __device__ __forceinline uint8_t BrendanCUDA::details::applyTargetFlips(uint8_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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

__global__ void BrendanCUDA::details::applyTargetFlipsOnArray_kernel(uint64_t* arr, uint32_t flipProb, uint64_t bs) {
    uint64_t& v(arr[blockIdx.x]);

    uint64_t rn64_1 = BrendanCUDA::Random::HashI64(BrendanCUDA::Random::GetSeedOnKernel(bs));
    uint64_t rn64_2 = BrendanCUDA::Random::HashI64(rn64_1 ^ 12210506935820558677);

    uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
    uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
    uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
    uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

    v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
}
__global__ void BrendanCUDA::details::applyTargetFlipsOnArray_kernel(uint32_t* arr, uint32_t flipProb, uint64_t bs) {
    uint32_t& v(arr[blockIdx.x]);

    uint64_t rn64_1 = BrendanCUDA::Random::HashI64(BrendanCUDA::Random::GetSeedOnKernel(bs));
    uint64_t rn64_2 = BrendanCUDA::Random::HashI64(rn64_1 ^ 484654973014905267);

    uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
    uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
    uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
    uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

    v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
}
__global__ void BrendanCUDA::details::applyTargetFlipsOnArray_kernel(uint16_t* arr, uint32_t flipProb, uint64_t bs) {
    uint16_t& v(arr[blockIdx.x]);

    uint64_t rn64_1 = BrendanCUDA::Random::HashI64(BrendanCUDA::Random::GetSeedOnKernel(bs));
    uint64_t rn64_2 = BrendanCUDA::Random::HashI64(rn64_1 ^ 3123193471197220784);

    uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
    uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
    uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
    uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

    v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
}
__global__ void BrendanCUDA::details::applyTargetFlipsOnArray_kernel(uint8_t* arr, uint32_t flipProb, uint64_t bs) {
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
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, RNG());
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
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, RNG());
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
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, RNG());
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
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, RNG());
#endif
}

template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::MLPBL() {
#ifdef __CUDA_ARCH__
    weights = new _TInput[sizeof(_TOutput) << 3];
    bias = new _TOutput;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(_TInput) * (sizeof(_TOutput) << 3)));
    ThrowIfBad(cudaMalloc(&bias, sizeof(_TOutput)));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::MLPBL(_TInput* Weights, _TOutput* Bias) {
    weights = Weights;
    bias = Bias;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <bool _CopyFromHost>
__host__ __forceinline auto BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CloneFrom(_TInput* Weights, _TOutput* Bias) -> MLPBL<_TInput, _TOutput> {
    _TInput* weights;
    _TOutput* bias;
    ThrowIfBad(cudaMalloc(&weights, sizeof(_TInput) * (sizeof(_TOutput) << 3)));
    ThrowIfBad(cudaMalloc(&bias, sizeof(_TOutput)));
    constexpr auto t = _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(_TOutput), t));
    return MLPBL(weights, bias);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CloneFrom(_TInput* Weights, _TOutput* Bias) -> MLPBL<_TInput, _TOutput> {
    _TInput* weights = new _TInput[sizeof(_TOutput) << 3];
    _TOutput* bias = new _TOutput;
    deviceMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
    *bias = *Bias;
    return MLPBL(weights, bias);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Dispose() {
#ifdef __CUDA_ARCH__
    delete[] weights;
    delete bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Weights() const {
    return weights;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Weight(size_t Index) const {
    return &weights[Index];
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TOutput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Bias() const {
    return bias;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <bool _CopyToHost>
__host__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyOutWeights() const {
    if (_CopyToHost) {
        _TInput* output = new _TInput[sizeof(_TOutput) << 3];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        _TInput* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(_TInput) * (sizeof(_TOutput) << 3)));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToDevice));
        return output;
    }
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <bool _CopyToHost>
__host__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyOutWeights(_TInput* Weights) const {
    cudaMemcpy(Weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), _CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__device__ __forceinline _TInput* BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyOutWeights() const {
    _TInput* output = new _TInput[sizeof(_TOutput) << 3];
    deviceMemcpy(output, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
    return output;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyOutWeights(_TInput* Weights) const {
    deviceMemcpy(Weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
template <bool _CopyFromHost>
__host__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyInWeights(_TInput* Weights) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyInWeights(_TInput* Weights) {
    deviceMemcpy(weights, Weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TInput BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::GetWeight(size_t Index) const {
#ifdef __CUDA_ARCH__
    return weights[Index];
#else
    _TInput output;
    ThrowIfBad(cudaMemcpy(&output, weights + Index, sizeof(_TInput), cudaMemcpyDeviceToHost));
    return output;
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::SetWeight(size_t Index, _TInput Weight) {
#ifdef __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(weights + Index, &Weight, sizeof(_TInput), cudaMemcpyHostToDevice));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TOutput BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::GetBias() const {
#ifdef __CUDA_ARCH__
    return *bias;
#else
    _TOutput output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost));
    return output;
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::SetBias(_TOutput Bias) {
#ifdef __CUDA_ARCH__
    * bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(_TOutput), cudaMemcpyHostToDevice));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline _TOutput BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Run(_TInput Input) const {
#ifdef __CUDA_ARCH__
    _TOutput o = 0;
    for (size_t i = 0; i < (sizeof(_TOutput) << 3); ++i) {
        if (Input & weights[i]) o |= ((_TOutput)1) << i;
    }
    return o ^ (*bias);
#else
    _TInput hWeights[sizeof(_TOutput) << 3];
    _TOutput o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < (sizeof(_TOutput) << 3); ++i) {
        if (Input & hWeights[i]) o |= ((_TOutput)1) << i;
    }
    return o;
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline uint64_t BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RunG(uint64_t Input) const {
    return (uint64_t)Run((_TInput)Input);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::CopyTo(MLPBL<_TInput, _TOutput> Other) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3));
    deviceMemcpy(Other.bias, bias, sizeof(_TOutput));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(_TOutput), cudaMemcpyDeviceToDevice));
#endif
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Clone() const {
    MLPBL<_TInput, _TOutput> n;
    CopyTo(n);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    RandomizeArray(Span<_TInput>(weights, sizeof(_TOutput) << 3), WeightsFlipProb, RNG);
    RandomizeArray(Span<_TOutput>(bias, 1), BiasFlipProb, RNG);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    MLPBL<_TInput, _TOutput> n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    details::applyTargetFlipsOnArray(weights, sizeof(_TOutput) << 3, WeightsEachFlipProb, RNG);
    RandomizeArray(Span<_TOutput>(bias, 1), BiasFlipProb, RNG);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    MLPBL<_TInput, _TOutput> n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) {
    for (uint32_t i = 0; i < (sizeof(_TOutput) << 3); ++i) {
        if (RNG() < WeightsMutationProb) {
            SetWeight(i, (_TInput)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, RNG));
        }
    }
    if (RNG() < WeightsMutationProb) {
        SetBias((_TOutput)BrendanCUDA::Random::Get64Bits(BiasProbOf1, RNG));
    }
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const {
    MLPBL<_TInput, _TOutput> n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
    return n;
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline size_t BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::SerializedSize() const {
    return sizeof(_TInput) * (sizeof(_TOutput) << 3) + sizeof(_TOutput);
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Serialize(void*& Data) const {
    if constexpr (std::endian::native == std::endian::little) {
        cudaMemcpy(Data, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost);
        Data = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        cudaMemcpy(Data, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost);
        Data = ((_TOutput*)Data) + 1;
    }
    else {
        cudaMemcpy(Data, weights, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyDeviceToHost);
        void* nd = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        BSerializer::ToFromLittleEndian((_TInput*)Data, (_TInput*)nd);
        Data = nd;
        cudaMemcpy(Data, bias, sizeof(_TOutput), cudaMemcpyDeviceToHost);
        _TOutput& bref = *(_TOutput*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((_TOutput*)Data) + 1;
    }
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput> BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Deserialize(const void*& Data) {
    if constexpr (std::endian::native == std::endian::little) {
        MLPBL<_TInput, _TOutput> mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyHostToDevice);
        Data = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        cudaMemcpy(mlpbl.bias, Data, sizeof(_TOutput), cudaMemcpyHostToDevice);
        Data = ((_TOutput*)Data) + 1;
        return mlpbl;
    }
    else {
        MLPBL mlpbl;
        cudaMemcpy(mlpbl.weights, Data, sizeof(_TInput) * (sizeof(_TOutput) << 3), cudaMemcpyHostToDevice);
        void* nd = ((_TInput*)Data) + (sizeof(_TOutput) << 3);
        BSerializer::ToFromLittleEndian((_TInput*)Data, (_TInput*)nd);
        Data = nd;
        cudaMemcpy(mlpbl.bias, Data, sizeof(_TOutput), cudaMemcpyHostToDevice);
        _TOutput& bref = *(_TOutput*)Data;
        bref = BSerializer::ToFromLittleEndian(bref);
        Data = ((_TOutput*)Data) + 1;
        return mlpbl;
    }
}
template <std::unsigned_integral _TInput, std::unsigned_integral _TOutput>
__forceinline void BrendanCUDA::AI::MLPB::MLPBL<_TInput, _TOutput>::Deserialize(const void*& Data, MLPBL<_TInput, _TOutput>& Value) {
    Value = Deserialize(Data);
}