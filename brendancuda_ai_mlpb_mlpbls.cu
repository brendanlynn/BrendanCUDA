#include "brendancuda_ai_mlpb_mlpbls.cuh"
#include "brendancuda_binary_basic.cuh"
#include "brendancuda_random_bits.cuh"
#include "brendancuda_cudaerrorhelpers.h"

__host__ __device__ uint64_t applyTargetFlipsTo1s_getEdits(uint64_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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
__host__ __device__ uint32_t applyTargetFlipsTo1s_getEdits(uint32_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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
__host__ __device__ uint16_t applyTargetFlipsTo1s_getEdits(uint16_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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
__host__ __device__ uint8_t applyTargetFlipsTo1s_getEdits(uint8_t Value, uint32_t CountOf1s, uint32_t FlipProb, uint32_t rn1, uint32_t rn2) {
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

__host__ __device__ uint64_t applyTargetFlips(uint64_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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
        uint32_t v1c = BrendanCUDA::Binary::Count1s(Value);

        return 
            Value ^
            applyTargetFlipsTo1s_getEdits(Value, v1c, FlipProb, rn1, rn2) ^
            applyTargetFlipsTo1s_getEdits(~Value, 64 - v1c, FlipProb, rn3, rn4);
    }
}
__host__ __device__ uint32_t applyTargetFlips(uint32_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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
        uint32_t v1c = BrendanCUDA::Binary::Count1s(Value);

        return 
            Value ^
            applyTargetFlipsTo1s_getEdits(Value, v1c, FlipProb, rn1, rn2) ^
            applyTargetFlipsTo1s_getEdits(~Value, 32 - v1c, FlipProb, rn3, rn4);
    }
}
__host__ __device__ uint16_t applyTargetFlips(uint16_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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
        uint32_t v1c = BrendanCUDA::Binary::Count1s(Value);

        return 
            Value ^
            applyTargetFlipsTo1s_getEdits(Value, v1c, FlipProb, rn1, rn2) ^
            applyTargetFlipsTo1s_getEdits((uint16_t)~Value, 16 - v1c, FlipProb, rn3, rn4);
    }
}
__host__ __device__ uint8_t applyTargetFlips(uint8_t Value, uint32_t FlipProb, uint32_t rn1, uint32_t rn2, uint32_t rn3, uint32_t rn4) {
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
        uint32_t v1c = BrendanCUDA::Binary::Count1s(Value);

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

__host__ __device__ void applyTargetFlipsOnArray(uint64_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> rng) {
#if __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint64_t& v(arr[i]);

        uint64_t rn64_1 = rng();
        uint64_t rn64_2 = rng();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, rng());
#endif
}
__host__ __device__ void applyTargetFlipsOnArray(uint32_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> rng) {
#if __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint32_t& v(arr[i]);

        uint64_t rn64_1 = rng();
        uint64_t rn64_2 = rng();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, rng());
#endif
}
__host__ __device__ void applyTargetFlipsOnArray(uint16_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> rng) {
#if __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint16_t& v(arr[i]);

        uint64_t rn64_1 = rng();
        uint64_t rn64_2 = rng();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, rng());
#endif
}
__host__ __device__ void applyTargetFlipsOnArray(uint8_t* arr, size_t sz, uint64_t flipProb, BrendanCUDA::Random::AnyRNG<uint64_t> rng) {
#if __CUDA_ARCH__
    for (size_t i = 0; i < sz; ++i) {
        uint8_t& v(arr[i]);

        uint64_t rn64_1 = rng();
        uint64_t rn64_2 = rng();

        uint32_t rn1 = ((uint32_t*)&rn64_1)[0];
        uint32_t rn2 = ((uint32_t*)&rn64_1)[1];
        uint32_t rn3 = ((uint32_t*)&rn64_2)[0];
        uint32_t rn4 = ((uint32_t*)&rn64_2)[1];

        v = applyTargetFlips(v, flipProb, rn1, rn2, rn3, rn4);
    }
#else
    applyTargetFlipsOnArray_kernel<<<sz, 1>>>(arr, flipProb, rng());
#endif
}

__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T8::MLPBL8T8() {
#if __CUDA_ARCH__
    weights = new uint8_t[8];
    bias = new uint8_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL8T8::MLPBL8T8(uint8_t* Weights, uint8_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 8, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint8_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL8T8::MLPBL8T8(uint8_t* Weights, uint8_t* Bias) {
    weights = new uint8_t[8];
    bias = new uint8_t;
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 8);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T8::MLPBL8T8(uint8_t* Weights, uint8_t Bias)
    : MLPBL8T8() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T8::Weights() const {
    return weights;
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T8::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T8::Bias() const {
    return bias;
}
__host__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T8::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint8_t* output = new uint8_t[8];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 8, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint8_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint8_t) * 8));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 8, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T8::GetWeights() const {
    uint8_t* output = new uint8_t[8];
    deviceMemcpy(output, weights, sizeof(uint8_t) * 8);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL8T8::SetWeights(uint8_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 8, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::SetWeights(uint8_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 8);
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL8T8::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::SetWeight(size_t Index, uint8_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL8T8::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::SetBias(uint8_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL8T8::Run(uint8_t Input) const {
#if __CUDA_ARCH__
    uint8_t o = 0i8;
    if (Input & weights[0])
        o |= 0b00000001;
    if (Input & weights[1])
        o |= 0b00000010;
    if (Input & weights[2])
        o |= 0b00000100;
    if (Input & weights[3])
        o |= 0b00001000;
    if (Input & weights[4])
        o |= 0b00010000;
    if (Input & weights[5])
        o |= 0b00100000;
    if (Input & weights[6])
        o |= 0b01000000;
    if (Input & weights[7])
        o |= 0b10000000;
    return o ^ (*bias);
#else
    uint8_t* hWeights = new uint8_t[8];
    uint8_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint8_t) * 8, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000001;
    if (Input & hWeights[1])
        o |= 0b00000010;
    if (Input & hWeights[2])
        o |= 0b00000100;
    if (Input & hWeights[3])
        o |= 0b00001000;
    if (Input & hWeights[4])
        o |= 0b00010000;
    if (Input & hWeights[5])
        o |= 0b00100000;
    if (Input & hWeights[6])
        o |= 0b01000000;
    if (Input & hWeights[7])
        o |= 0b10000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL8T8::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint8_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::CopyTo(MLPBL8T8 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint8_t) * 8);
    deviceMemcpy(Other.bias, bias, sizeof(uint8_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint8_t) * 8, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint8_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T8 BrendanCUDA::AI::MLPB::MLPBL8T8::Clone() const {
    MLPBL8T8 n = MLPBL8T8();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T8 BrendanCUDA::AI::MLPB::MLPBL8T8::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T8 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T8 BrendanCUDA::AI::MLPB::MLPBL8T8::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T8 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T8::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint8_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint8_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T8 BrendanCUDA::AI::MLPB::MLPBL8T8::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T8 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL8T8::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL8T8 BrendanCUDA::AI::MLPB::MLPBL8T8::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T16::MLPBL8T16() {
#if __CUDA_ARCH__
    weights = new uint8_t[16];
    bias = new uint16_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL8T16::MLPBL8T16(uint8_t* Weights, uint16_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 16, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint16_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL8T16::MLPBL8T16(uint8_t* Weights, uint16_t* Bias) {
    weights = new uint8_t[16];
    bias = new uint16_t;
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 16);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T16::MLPBL8T16(uint8_t* Weights, uint16_t Bias)
    : MLPBL8T16() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T16::Weights() const {
    return weights;
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T16::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL8T16::Bias() const {
    return bias;
}
__host__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T16::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint8_t* output = new uint8_t[16];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 16, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint8_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint16_t) * 16));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 16, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T16::GetWeights() const {
    uint8_t* output = new uint8_t[16];
    deviceMemcpy(output, weights, sizeof(uint8_t) * 16);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL8T16::SetWeights(uint8_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 16, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::SetWeights(uint8_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 16);
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL8T16::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::SetWeight(size_t Index, uint8_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL8T16::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::SetBias(uint16_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL8T16::Run(uint8_t Input) const {
#if __CUDA_ARCH__
    uint16_t o = 0i16;
    if (Input & weights[0])
        o |= 0b00000001;
    if (Input & weights[1])
        o |= 0b00000010;
    if (Input & weights[2])
        o |= 0b00000100;
    if (Input & weights[3])
        o |= 0b00001000;
    if (Input & weights[4])
        o |= 0b00010000;
    if (Input & weights[5])
        o |= 0b00100000;
    if (Input & weights[6])
        o |= 0b01000000;
    if (Input & weights[7])
        o |= 0b10000000;
    return o ^ (*bias);
#else
    uint16_t* hWeights = new uint16_t[8];
    uint16_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint16_t) * 8, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000001;
    if (Input & hWeights[1])
        o |= 0b00000010;
    if (Input & hWeights[2])
        o |= 0b00000100;
    if (Input & hWeights[3])
        o |= 0b00001000;
    if (Input & hWeights[4])
        o |= 0b00010000;
    if (Input & hWeights[5])
        o |= 0b00100000;
    if (Input & hWeights[6])
        o |= 0b01000000;
    if (Input & hWeights[7])
        o |= 0b10000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL8T16::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint8_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::CopyTo(MLPBL8T16 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint8_t) * 16);
    deviceMemcpy(Other.bias, bias, sizeof(uint16_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint8_t) * 16, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint16_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T16 BrendanCUDA::AI::MLPB::MLPBL8T16::Clone() const {
    MLPBL8T16 n = MLPBL8T16();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T16 BrendanCUDA::AI::MLPB::MLPBL8T16::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T16 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T16 BrendanCUDA::AI::MLPB::MLPBL8T16::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T16 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T16::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint8_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint16_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T16 BrendanCUDA::AI::MLPB::MLPBL8T16::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T16 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL8T16::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL8T16 BrendanCUDA::AI::MLPB::MLPBL8T16::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T32::MLPBL8T32() {
#if __CUDA_ARCH__
    weights = new uint8_t[32];
    bias = new uint32_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL8T32::MLPBL8T32(uint8_t* Weights, uint32_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 32, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint32_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL8T32::MLPBL8T32(uint8_t* Weights, uint32_t* Bias) {
    weights = new uint8_t[32];
    bias = new uint32_t;
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 32);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T32::MLPBL8T32(uint8_t* Weights, uint32_t Bias)
    : MLPBL8T32() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T32::Weights() const {
    return weights;
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T32::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL8T32::Bias() const {
    return bias;
}
__host__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T32::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint8_t* output = new uint8_t[32];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 32, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint8_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint32_t) * 32));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 32, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T32::GetWeights() const {
    uint8_t* output = new uint8_t[32];
    deviceMemcpy(output, weights, sizeof(uint8_t) * 32);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL8T32::SetWeights(uint8_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 32, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::SetWeights(uint8_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 32);
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL8T32::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::SetWeight(size_t Index, uint8_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL8T32::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::SetBias(uint32_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL8T32::Run(uint8_t Input) const {
#if __CUDA_ARCH__
    uint32_t o = 0i32;
    if (Input & weights[0])
        o |= 0b00000001;
    if (Input & weights[1])
        o |= 0b00000010;
    if (Input & weights[2])
        o |= 0b00000100;
    if (Input & weights[3])
        o |= 0b00001000;
    if (Input & weights[4])
        o |= 0b00010000;
    if (Input & weights[5])
        o |= 0b00100000;
    if (Input & weights[6])
        o |= 0b01000000;
    if (Input & weights[7])
        o |= 0b10000000;
    return o ^ (*bias);
#else
    uint32_t* hWeights = new uint32_t[8];
    uint32_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000001;
    if (Input & hWeights[1])
        o |= 0b00000010;
    if (Input & hWeights[2])
        o |= 0b00000100;
    if (Input & hWeights[3])
        o |= 0b00001000;
    if (Input & hWeights[4])
        o |= 0b00010000;
    if (Input & hWeights[5])
        o |= 0b00100000;
    if (Input & hWeights[6])
        o |= 0b01000000;
    if (Input & hWeights[7])
        o |= 0b10000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL8T32::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint8_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::CopyTo(MLPBL8T32 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint8_t) * 32);
    deviceMemcpy(Other.bias, bias, sizeof(uint32_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint8_t) * 32, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T32 BrendanCUDA::AI::MLPB::MLPBL8T32::Clone() const {
    MLPBL8T32 n = MLPBL8T32();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T32 BrendanCUDA::AI::MLPB::MLPBL8T32::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T32 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T32 BrendanCUDA::AI::MLPB::MLPBL8T32::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T32 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T32::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint8_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint32_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T32 BrendanCUDA::AI::MLPB::MLPBL8T32::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T32 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL8T32::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL8T32 BrendanCUDA::AI::MLPB::MLPBL8T32::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T64::MLPBL8T64() {
#if __CUDA_ARCH__
    weights = new uint8_t[64];
    bias = new uint64_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL8T64::MLPBL8T64(uint8_t* Weights, uint64_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint8_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 64, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint64_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL8T64::MLPBL8T64(uint8_t* Weights, uint64_t* Bias) {
    weights = new uint8_t[64];
    bias = new uint64_t;
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 64);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T64::MLPBL8T64(uint8_t* Weights, uint64_t Bias)
    : MLPBL8T64() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T64::Weights() const {
    return weights;
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T64::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL8T64::Bias() const {
    return bias;
}
__host__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T64::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint8_t* output = new uint8_t[64];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 64, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint8_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint64_t) * 64));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint8_t) * 64, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL8T64::GetWeights() const {
    uint8_t* output = new uint8_t[64];
    deviceMemcpy(output, weights, sizeof(uint8_t) * 64);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL8T64::SetWeights(uint8_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint8_t) * 64, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::SetWeights(uint8_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint8_t) * 64);
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL8T64::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::SetWeight(size_t Index, uint8_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL8T64::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::SetBias(uint64_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL8T64::Run(uint8_t Input) const {
#if __CUDA_ARCH__
    uint64_t o = 0i64;
    if (Input & weights[0])
        o |= 0b00000001;
    if (Input & weights[1])
        o |= 0b00000010;
    if (Input & weights[2])
        o |= 0b00000100;
    if (Input & weights[3])
        o |= 0b00001000;
    if (Input & weights[4])
        o |= 0b00010000;
    if (Input & weights[5])
        o |= 0b00100000;
    if (Input & weights[6])
        o |= 0b01000000;
    if (Input & weights[7])
        o |= 0b10000000;
    return o ^ (*bias);
#else
    uint64_t* hWeights = new uint64_t[8];
    uint64_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint64_t) * 8, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000001;
    if (Input & hWeights[1])
        o |= 0b00000010;
    if (Input & hWeights[2])
        o |= 0b00000100;
    if (Input & hWeights[3])
        o |= 0b00001000;
    if (Input & hWeights[4])
        o |= 0b00010000;
    if (Input & hWeights[5])
        o |= 0b00100000;
    if (Input & hWeights[6])
        o |= 0b01000000;
    if (Input & hWeights[7])
        o |= 0b10000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL8T64::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint8_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::CopyTo(MLPBL8T64 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint8_t) * 64);
    deviceMemcpy(Other.bias, bias, sizeof(uint64_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint8_t) * 64, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint64_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T64 BrendanCUDA::AI::MLPB::MLPBL8T64::Clone() const {
    MLPBL8T64 n = MLPBL8T64();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T64 BrendanCUDA::AI::MLPB::MLPBL8T64::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T64 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T64 BrendanCUDA::AI::MLPB::MLPBL8T64::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T64 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL8T64::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint8_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint64_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL8T64 BrendanCUDA::AI::MLPB::MLPBL8T64::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL8T64 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL8T64::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL8T64 BrendanCUDA::AI::MLPB::MLPBL8T64::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T8::MLPBL16T8() {
#if __CUDA_ARCH__
    weights = new uint16_t[8];
    bias = new uint8_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL16T8::MLPBL16T8(uint16_t* Weights, uint8_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 8, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint8_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL16T8::MLPBL16T8(uint16_t* Weights, uint8_t* Bias) {
    weights = new uint16_t[8];
    bias = new uint8_t;
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 8);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T8::MLPBL16T8(uint16_t* Weights, uint8_t Bias)
    : MLPBL16T8() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T8::Weights() const {
    return weights;
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T8::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL16T8::Bias() const {
    return bias;
}
__host__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T8::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint16_t* output = new uint16_t[8];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 8, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint16_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint8_t) * 8));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 8, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T8::GetWeights() const {
    uint16_t* output = new uint16_t[8];
    deviceMemcpy(output, weights, sizeof(uint16_t) * 8);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL16T8::SetWeights(uint16_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 8, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::SetWeights(uint16_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 8);
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL16T8::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::SetWeight(size_t Index, uint16_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL16T8::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::SetBias(uint8_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL16T8::Run(uint16_t Input) const {
#if __CUDA_ARCH__
    uint8_t o = 0i8;
    if (Input & weights[0])
        o |= 0b0000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000100000;
    if (Input & weights[6])
        o |= 0b0000000001000000;
    if (Input & weights[7])
        o |= 0b0000000010000000;
    if (Input & weights[8])
        o |= 0b0000000100000000;
    if (Input & weights[9])
        o |= 0b0000001000000000;
    if (Input & weights[10])
        o |= 0b0000010000000000;
    if (Input & weights[11])
        o |= 0b0000100000000000;
    if (Input & weights[12])
        o |= 0b0001000000000000;
    if (Input & weights[13])
        o |= 0b0010000000000000;
    if (Input & weights[14])
        o |= 0b0100000000000000;
    if (Input & weights[15])
        o |= 0b1000000000000000;
    return o ^ (*bias);
#else
    uint8_t* hWeights = new uint8_t[16];
    uint8_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint8_t) * 16, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000100000000000;
    if (Input & hWeights[12])
        o |= 0b0001000000000000;
    if (Input & hWeights[13])
        o |= 0b0010000000000000;
    if (Input & hWeights[14])
        o |= 0b0100000000000000;
    if (Input & hWeights[15])
        o |= 0b1000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL16T8::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint16_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::CopyTo(MLPBL16T8 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint16_t) * 8);
    deviceMemcpy(Other.bias, bias, sizeof(uint8_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint16_t) * 8, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint8_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T8 BrendanCUDA::AI::MLPB::MLPBL16T8::Clone() const {
    MLPBL16T8 n = MLPBL16T8();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T8 BrendanCUDA::AI::MLPB::MLPBL16T8::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T8 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T8 BrendanCUDA::AI::MLPB::MLPBL16T8::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T8 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T8::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint16_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint8_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T8 BrendanCUDA::AI::MLPB::MLPBL16T8::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T8 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL16T8::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL16T8 BrendanCUDA::AI::MLPB::MLPBL16T8::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T16::MLPBL16T16() {
#if __CUDA_ARCH__
    weights = new uint16_t[16];
    bias = new uint16_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL16T16::MLPBL16T16(uint16_t* Weights, uint16_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 16, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint16_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL16T16::MLPBL16T16(uint16_t* Weights, uint16_t* Bias) {
    weights = new uint16_t[16];
    bias = new uint16_t;
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 16);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T16::MLPBL16T16(uint16_t* Weights, uint16_t Bias)
    : MLPBL16T16() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T16::Weights() const {
    return weights;
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T16::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T16::Bias() const {
    return bias;
}
__host__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T16::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint16_t* output = new uint16_t[16];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 16, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint16_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint16_t) * 16));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 16, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T16::GetWeights() const {
    uint16_t* output = new uint16_t[16];
    deviceMemcpy(output, weights, sizeof(uint16_t) * 16);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL16T16::SetWeights(uint16_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 16, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::SetWeights(uint16_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 16);
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL16T16::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::SetWeight(size_t Index, uint16_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL16T16::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::SetBias(uint16_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL16T16::Run(uint16_t Input) const {
#if __CUDA_ARCH__
    uint16_t o = 0i16;
    if (Input & weights[0])
        o |= 0b0000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000100000;
    if (Input & weights[6])
        o |= 0b0000000001000000;
    if (Input & weights[7])
        o |= 0b0000000010000000;
    if (Input & weights[8])
        o |= 0b0000000100000000;
    if (Input & weights[9])
        o |= 0b0000001000000000;
    if (Input & weights[10])
        o |= 0b0000010000000000;
    if (Input & weights[11])
        o |= 0b0000100000000000;
    if (Input & weights[12])
        o |= 0b0001000000000000;
    if (Input & weights[13])
        o |= 0b0010000000000000;
    if (Input & weights[14])
        o |= 0b0100000000000000;
    if (Input & weights[15])
        o |= 0b1000000000000000;
    return o ^ (*bias);
#else
    uint16_t* hWeights = new uint16_t[16];
    uint16_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint16_t) * 16, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000100000000000;
    if (Input & hWeights[12])
        o |= 0b0001000000000000;
    if (Input & hWeights[13])
        o |= 0b0010000000000000;
    if (Input & hWeights[14])
        o |= 0b0100000000000000;
    if (Input & hWeights[15])
        o |= 0b1000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL16T16::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint16_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::CopyTo(MLPBL16T16 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint16_t) * 16);
    deviceMemcpy(Other.bias, bias, sizeof(uint16_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint16_t) * 16, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint16_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T16 BrendanCUDA::AI::MLPB::MLPBL16T16::Clone() const {
    MLPBL16T16 n = MLPBL16T16();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T16 BrendanCUDA::AI::MLPB::MLPBL16T16::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T16 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T16 BrendanCUDA::AI::MLPB::MLPBL16T16::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T16 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T16::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint16_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint16_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T16 BrendanCUDA::AI::MLPB::MLPBL16T16::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T16 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL16T16::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL16T16 BrendanCUDA::AI::MLPB::MLPBL16T16::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T32::MLPBL16T32() {
#if __CUDA_ARCH__
    weights = new uint16_t[32];
    bias = new uint32_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL16T32::MLPBL16T32(uint16_t* Weights, uint32_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 32, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint32_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL16T32::MLPBL16T32(uint16_t* Weights, uint32_t* Bias) {
    weights = new uint16_t[32];
    bias = new uint32_t;
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 32);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T32::MLPBL16T32(uint16_t* Weights, uint32_t Bias)
    : MLPBL16T32() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T32::Weights() const {
    return weights;
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T32::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL16T32::Bias() const {
    return bias;
}
__host__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T32::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint16_t* output = new uint16_t[32];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 32, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint16_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint32_t) * 32));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 32, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T32::GetWeights() const {
    uint16_t* output = new uint16_t[32];
    deviceMemcpy(output, weights, sizeof(uint16_t) * 32);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL16T32::SetWeights(uint16_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 32, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::SetWeights(uint16_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 32);
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL16T32::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::SetWeight(size_t Index, uint16_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL16T32::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::SetBias(uint32_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL16T32::Run(uint16_t Input) const {
#if __CUDA_ARCH__
    uint32_t o = 0i32;
    if (Input & weights[0])
        o |= 0b0000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000100000;
    if (Input & weights[6])
        o |= 0b0000000001000000;
    if (Input & weights[7])
        o |= 0b0000000010000000;
    if (Input & weights[8])
        o |= 0b0000000100000000;
    if (Input & weights[9])
        o |= 0b0000001000000000;
    if (Input & weights[10])
        o |= 0b0000010000000000;
    if (Input & weights[11])
        o |= 0b0000100000000000;
    if (Input & weights[12])
        o |= 0b0001000000000000;
    if (Input & weights[13])
        o |= 0b0010000000000000;
    if (Input & weights[14])
        o |= 0b0100000000000000;
    if (Input & weights[15])
        o |= 0b1000000000000000;
    return o ^ (*bias);
#else
    uint32_t* hWeights = new uint32_t[16];
    uint32_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000100000000000;
    if (Input & hWeights[12])
        o |= 0b0001000000000000;
    if (Input & hWeights[13])
        o |= 0b0010000000000000;
    if (Input & hWeights[14])
        o |= 0b0100000000000000;
    if (Input & hWeights[15])
        o |= 0b1000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL16T32::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint16_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::CopyTo(MLPBL16T32 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint16_t) * 32);
    deviceMemcpy(Other.bias, bias, sizeof(uint32_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint16_t) * 32, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T32 BrendanCUDA::AI::MLPB::MLPBL16T32::Clone() const {
    MLPBL16T32 n = MLPBL16T32();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T32 BrendanCUDA::AI::MLPB::MLPBL16T32::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T32 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T32 BrendanCUDA::AI::MLPB::MLPBL16T32::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T32 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T32::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint16_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint32_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T32 BrendanCUDA::AI::MLPB::MLPBL16T32::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T32 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL16T32::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL16T32 BrendanCUDA::AI::MLPB::MLPBL16T32::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T64::MLPBL16T64() {
#if __CUDA_ARCH__
    weights = new uint16_t[64];
    bias = new uint64_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL16T64::MLPBL16T64(uint16_t* Weights, uint64_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint16_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 64, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint64_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL16T64::MLPBL16T64(uint16_t* Weights, uint64_t* Bias) {
    weights = new uint16_t[64];
    bias = new uint64_t;
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 64);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T64::MLPBL16T64(uint16_t* Weights, uint64_t Bias)
    : MLPBL16T64() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T64::Weights() const {
    return weights;
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T64::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL16T64::Bias() const {
    return bias;
}
__host__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T64::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint16_t* output = new uint16_t[64];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 64, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint16_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint64_t) * 64));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint16_t) * 64, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL16T64::GetWeights() const {
    uint16_t* output = new uint16_t[64];
    deviceMemcpy(output, weights, sizeof(uint16_t) * 64);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL16T64::SetWeights(uint16_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint16_t) * 64, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::SetWeights(uint16_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint16_t) * 64);
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL16T64::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::SetWeight(size_t Index, uint16_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL16T64::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::SetBias(uint64_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL16T64::Run(uint16_t Input) const {
#if __CUDA_ARCH__
    uint64_t o = 0i64;
    if (Input & weights[0])
        o |= 0b0000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000100000;
    if (Input & weights[6])
        o |= 0b0000000001000000;
    if (Input & weights[7])
        o |= 0b0000000010000000;
    if (Input & weights[8])
        o |= 0b0000000100000000;
    if (Input & weights[9])
        o |= 0b0000001000000000;
    if (Input & weights[10])
        o |= 0b0000010000000000;
    if (Input & weights[11])
        o |= 0b0000100000000000;
    if (Input & weights[12])
        o |= 0b0001000000000000;
    if (Input & weights[13])
        o |= 0b0010000000000000;
    if (Input & weights[14])
        o |= 0b0100000000000000;
    if (Input & weights[15])
        o |= 0b1000000000000000;
    return o ^ (*bias);
#else
    uint64_t* hWeights = new uint64_t[16];
    uint64_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint64_t) * 16, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000100000000000;
    if (Input & hWeights[12])
        o |= 0b0001000000000000;
    if (Input & hWeights[13])
        o |= 0b0010000000000000;
    if (Input & hWeights[14])
        o |= 0b0100000000000000;
    if (Input & hWeights[15])
        o |= 0b1000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL16T64::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint16_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::CopyTo(MLPBL16T64 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint16_t) * 64);
    deviceMemcpy(Other.bias, bias, sizeof(uint64_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint16_t) * 64, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint64_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T64 BrendanCUDA::AI::MLPB::MLPBL16T64::Clone() const {
    MLPBL16T64 n = MLPBL16T64();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T64 BrendanCUDA::AI::MLPB::MLPBL16T64::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T64 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T64 BrendanCUDA::AI::MLPB::MLPBL16T64::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T64 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL16T64::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint16_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint64_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL16T64 BrendanCUDA::AI::MLPB::MLPBL16T64::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL16T64 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL16T64::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL16T64 BrendanCUDA::AI::MLPB::MLPBL16T64::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T8::MLPBL32T8() {
#if __CUDA_ARCH__
    weights = new uint32_t[8];
    bias = new uint8_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL32T8::MLPBL32T8(uint32_t* Weights, uint8_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 8, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint8_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL32T8::MLPBL32T8(uint32_t* Weights, uint8_t* Bias) {
    weights = new uint32_t[8];
    bias = new uint8_t;
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 8);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T8::MLPBL32T8(uint32_t* Weights, uint8_t Bias)
    : MLPBL32T8() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T8::Weights() const {
    return weights;
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T8::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL32T8::Bias() const {
    return bias;
}
__host__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T8::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint32_t* output = new uint32_t[8];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint32_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint8_t) * 8));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 8, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T8::GetWeights() const {
    uint32_t* output = new uint32_t[8];
    deviceMemcpy(output, weights, sizeof(uint32_t) * 8);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL32T8::SetWeights(uint32_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 8, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::SetWeights(uint32_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 8);
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL32T8::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::SetWeight(size_t Index, uint32_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL32T8::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::SetBias(uint8_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL32T8::Run(uint32_t Input) const {
#if __CUDA_ARCH__
    uint8_t o = 0i8;
    if (Input & weights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b10000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint8_t* hWeights = new uint8_t[32];
    uint8_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint8_t) * 32, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b10000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL32T8::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint32_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::CopyTo(MLPBL32T8 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint32_t) * 8);
    deviceMemcpy(Other.bias, bias, sizeof(uint8_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint32_t) * 8, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint8_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T8 BrendanCUDA::AI::MLPB::MLPBL32T8::Clone() const {
    MLPBL32T8 n = MLPBL32T8();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T8 BrendanCUDA::AI::MLPB::MLPBL32T8::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T8 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T8 BrendanCUDA::AI::MLPB::MLPBL32T8::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T8 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T8::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint32_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint8_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T8 BrendanCUDA::AI::MLPB::MLPBL32T8::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T8 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL32T8::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL32T8 BrendanCUDA::AI::MLPB::MLPBL32T8::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T16::MLPBL32T16() {
#if __CUDA_ARCH__
    weights = new uint32_t[16];
    bias = new uint16_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL32T16::MLPBL32T16(uint32_t* Weights, uint16_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 16, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint16_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL32T16::MLPBL32T16(uint32_t* Weights, uint16_t* Bias) {
    weights = new uint32_t[16];
    bias = new uint16_t;
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 16);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T16::MLPBL32T16(uint32_t* Weights, uint16_t Bias)
    : MLPBL32T16() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T16::Weights() const {
    return weights;
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T16::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL32T16::Bias() const {
    return bias;
}
__host__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T16::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint32_t* output = new uint32_t[16];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint32_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint16_t) * 16));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 16, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T16::GetWeights() const {
    uint32_t* output = new uint32_t[16];
    deviceMemcpy(output, weights, sizeof(uint32_t) * 16);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL32T16::SetWeights(uint32_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 16, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::SetWeights(uint32_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 16);
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL32T16::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::SetWeight(size_t Index, uint32_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL32T16::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::SetBias(uint16_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL32T16::Run(uint32_t Input) const {
#if __CUDA_ARCH__
    uint16_t o = 0i16;
    if (Input & weights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b10000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint16_t* hWeights = new uint16_t[32];
    uint16_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint16_t) * 32, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b10000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL32T16::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint32_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::CopyTo(MLPBL32T16 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint32_t) * 16);
    deviceMemcpy(Other.bias, bias, sizeof(uint16_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint32_t) * 16, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint16_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T16 BrendanCUDA::AI::MLPB::MLPBL32T16::Clone() const {
    MLPBL32T16 n = MLPBL32T16();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T16 BrendanCUDA::AI::MLPB::MLPBL32T16::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T16 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T16 BrendanCUDA::AI::MLPB::MLPBL32T16::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T16 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T16::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint32_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint16_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T16 BrendanCUDA::AI::MLPB::MLPBL32T16::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T16 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL32T16::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL32T16 BrendanCUDA::AI::MLPB::MLPBL32T16::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T32::MLPBL32T32() {
#if __CUDA_ARCH__
    weights = new uint32_t[32];
    bias = new uint32_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL32T32::MLPBL32T32(uint32_t* Weights, uint32_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 32, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint32_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL32T32::MLPBL32T32(uint32_t* Weights, uint32_t* Bias) {
    weights = new uint32_t[32];
    bias = new uint32_t;
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 32);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T32::MLPBL32T32(uint32_t* Weights, uint32_t Bias)
    : MLPBL32T32() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T32::Weights() const {
    return weights;
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T32::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T32::Bias() const {
    return bias;
}
__host__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T32::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint32_t* output = new uint32_t[32];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 32, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint32_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint32_t) * 32));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 32, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T32::GetWeights() const {
    uint32_t* output = new uint32_t[32];
    deviceMemcpy(output, weights, sizeof(uint32_t) * 32);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL32T32::SetWeights(uint32_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 32, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::SetWeights(uint32_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 32);
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL32T32::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::SetWeight(size_t Index, uint32_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL32T32::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::SetBias(uint32_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL32T32::Run(uint32_t Input) const {
#if __CUDA_ARCH__
    uint32_t o = 0i32;
    if (Input & weights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b10000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint32_t* hWeights = new uint32_t[32];
    uint32_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint32_t) * 32, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b10000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL32T32::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint32_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::CopyTo(MLPBL32T32 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint32_t) * 32);
    deviceMemcpy(Other.bias, bias, sizeof(uint32_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint32_t) * 32, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T32 BrendanCUDA::AI::MLPB::MLPBL32T32::Clone() const {
    MLPBL32T32 n = MLPBL32T32();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T32 BrendanCUDA::AI::MLPB::MLPBL32T32::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T32 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T32 BrendanCUDA::AI::MLPB::MLPBL32T32::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T32 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T32::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint32_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint32_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T32 BrendanCUDA::AI::MLPB::MLPBL32T32::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T32 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL32T32::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL32T32 BrendanCUDA::AI::MLPB::MLPBL32T32::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T64::MLPBL32T64() {
#if __CUDA_ARCH__
    weights = new uint32_t[64];
    bias = new uint64_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL32T64::MLPBL32T64(uint32_t* Weights, uint64_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint32_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 64, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint64_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL32T64::MLPBL32T64(uint32_t* Weights, uint64_t* Bias) {
    weights = new uint32_t[64];
    bias = new uint64_t;
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 64);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T64::MLPBL32T64(uint32_t* Weights, uint64_t Bias)
    : MLPBL32T64() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T64::Weights() const {
    return weights;
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T64::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL32T64::Bias() const {
    return bias;
}
__host__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T64::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint32_t* output = new uint32_t[64];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 64, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint32_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint64_t) * 64));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint32_t) * 64, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL32T64::GetWeights() const {
    uint32_t* output = new uint32_t[64];
    deviceMemcpy(output, weights, sizeof(uint32_t) * 64);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL32T64::SetWeights(uint32_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint32_t) * 64, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::SetWeights(uint32_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint32_t) * 64);
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL32T64::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::SetWeight(size_t Index, uint32_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL32T64::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::SetBias(uint64_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL32T64::Run(uint32_t Input) const {
#if __CUDA_ARCH__
    uint64_t o = 0i64;
    if (Input & weights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b10000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint64_t* hWeights = new uint64_t[32];
    uint64_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint64_t) * 32, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b00000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b00000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b00000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b00000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b00000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b00000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b00000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b00000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b00000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b00000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b00000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b00000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b00000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b00000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b00000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b00000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b00000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b00000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b00000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b00000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b00000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b00000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b00000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b00000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b00000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b00000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b00000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b00001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b00010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b00100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b01000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b10000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL32T64::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint32_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::CopyTo(MLPBL32T64 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint32_t) * 64);
    deviceMemcpy(Other.bias, bias, sizeof(uint64_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint32_t) * 64, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint64_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T64 BrendanCUDA::AI::MLPB::MLPBL32T64::Clone() const {
    MLPBL32T64 n = MLPBL32T64();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T64 BrendanCUDA::AI::MLPB::MLPBL32T64::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T64 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T64 BrendanCUDA::AI::MLPB::MLPBL32T64::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T64 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL32T64::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint32_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint64_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL32T64 BrendanCUDA::AI::MLPB::MLPBL32T64::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL32T64 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL32T64::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL32T64 BrendanCUDA::AI::MLPB::MLPBL32T64::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T8::MLPBL64T8() {
#if __CUDA_ARCH__
    weights = new uint64_t[8];
    bias = new uint8_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL64T8::MLPBL64T8(uint64_t* Weights, uint8_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 8));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint8_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 8, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint8_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL64T8::MLPBL64T8(uint64_t* Weights, uint8_t* Bias) {
    weights = new uint64_t[8];
    bias = new uint8_t;
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 8);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T8::MLPBL64T8(uint64_t* Weights, uint8_t Bias)
    : MLPBL64T8() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T8::Weights() const {
    return weights;
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T8::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint8_t* BrendanCUDA::AI::MLPB::MLPBL64T8::Bias() const {
    return bias;
}
__host__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T8::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint64_t* output = new uint64_t[8];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 8, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint64_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint8_t) * 8));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 8, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T8::GetWeights() const {
    uint64_t* output = new uint64_t[8];
    deviceMemcpy(output, weights, sizeof(uint64_t) * 8);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL64T8::SetWeights(uint64_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 8, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::SetWeights(uint64_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 8);
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T8::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::SetWeight(size_t Index, uint64_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL64T8::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint8_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::SetBias(uint8_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint8_t BrendanCUDA::AI::MLPB::MLPBL64T8::Run(uint64_t Input) const {
#if __CUDA_ARCH__
    uint8_t o = 0i8;
    if (Input & weights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & weights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & weights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & weights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & weights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & weights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & weights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & weights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & weights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & weights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & weights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & weights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & weights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & weights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & weights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & weights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & weights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & weights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & weights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & weights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & weights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & weights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & weights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & weights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & weights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & weights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & weights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint8_t* hWeights = new uint8_t[64];
    uint8_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint8_t) * 64, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & hWeights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & hWeights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & hWeights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & hWeights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & hWeights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & hWeights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & hWeights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & hWeights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & hWeights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & hWeights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & hWeights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & hWeights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & hWeights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & hWeights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & hWeights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & hWeights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & hWeights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & hWeights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & hWeights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & hWeights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T8::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint64_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::CopyTo(MLPBL64T8 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint64_t) * 8);
    deviceMemcpy(Other.bias, bias, sizeof(uint8_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint64_t) * 8, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint8_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T8 BrendanCUDA::AI::MLPB::MLPBL64T8::Clone() const {
    MLPBL64T8 n = MLPBL64T8();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T8 BrendanCUDA::AI::MLPB::MLPBL64T8::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T8 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T8 BrendanCUDA::AI::MLPB::MLPBL64T8::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T8 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T8::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint64_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint8_t v = (uint8_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint8_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T8 BrendanCUDA::AI::MLPB::MLPBL64T8::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T8 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL64T8::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL64T8 BrendanCUDA::AI::MLPB::MLPBL64T8::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T16::MLPBL64T16() {
#if __CUDA_ARCH__
    weights = new uint64_t[16];
    bias = new uint16_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL64T16::MLPBL64T16(uint64_t* Weights, uint16_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 16));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint16_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 16, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint16_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL64T16::MLPBL64T16(uint64_t* Weights, uint16_t* Bias) {
    weights = new uint64_t[16];
    bias = new uint16_t;
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 16);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T16::MLPBL64T16(uint64_t* Weights, uint16_t Bias)
    : MLPBL64T16() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T16::Weights() const {
    return weights;
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T16::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint16_t* BrendanCUDA::AI::MLPB::MLPBL64T16::Bias() const {
    return bias;
}
__host__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T16::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint64_t* output = new uint64_t[16];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 16, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint64_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint16_t) * 16));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 16, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T16::GetWeights() const {
    uint64_t* output = new uint64_t[16];
    deviceMemcpy(output, weights, sizeof(uint64_t) * 16);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL64T16::SetWeights(uint64_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 16, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::SetWeights(uint64_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 16);
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T16::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::SetWeight(size_t Index, uint64_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL64T16::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint16_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::SetBias(uint16_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint16_t BrendanCUDA::AI::MLPB::MLPBL64T16::Run(uint64_t Input) const {
#if __CUDA_ARCH__
    uint16_t o = 0i16;
    if (Input & weights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & weights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & weights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & weights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & weights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & weights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & weights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & weights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & weights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & weights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & weights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & weights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & weights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & weights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & weights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & weights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & weights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & weights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & weights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & weights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & weights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & weights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & weights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & weights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & weights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & weights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & weights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint16_t* hWeights = new uint16_t[64];
    uint16_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint16_t) * 64, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & hWeights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & hWeights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & hWeights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & hWeights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & hWeights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & hWeights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & hWeights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & hWeights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & hWeights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & hWeights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & hWeights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & hWeights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & hWeights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & hWeights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & hWeights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & hWeights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & hWeights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & hWeights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & hWeights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & hWeights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T16::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint64_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::CopyTo(MLPBL64T16 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint64_t) * 16);
    deviceMemcpy(Other.bias, bias, sizeof(uint16_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint64_t) * 16, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint16_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T16 BrendanCUDA::AI::MLPB::MLPBL64T16::Clone() const {
    MLPBL64T16 n = MLPBL64T16();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T16 BrendanCUDA::AI::MLPB::MLPBL64T16::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T16 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T16 BrendanCUDA::AI::MLPB::MLPBL64T16::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T16 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T16::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint64_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint16_t v = (uint16_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint16_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint16_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T16 BrendanCUDA::AI::MLPB::MLPBL64T16::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T16 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL64T16::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL64T16 BrendanCUDA::AI::MLPB::MLPBL64T16::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T32::MLPBL64T32() {
#if __CUDA_ARCH__
    weights = new uint64_t[32];
    bias = new uint32_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL64T32::MLPBL64T32(uint64_t* Weights, uint32_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 32));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint32_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 32, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint32_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL64T32::MLPBL64T32(uint64_t* Weights, uint32_t* Bias) {
    weights = new uint64_t[32];
    bias = new uint32_t;
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 32);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T32::MLPBL64T32(uint64_t* Weights, uint32_t Bias)
    : MLPBL64T32() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T32::Weights() const {
    return weights;
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T32::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint32_t* BrendanCUDA::AI::MLPB::MLPBL64T32::Bias() const {
    return bias;
}
__host__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T32::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint64_t* output = new uint64_t[32];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 32, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint64_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint32_t) * 32));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 32, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T32::GetWeights() const {
    uint64_t* output = new uint64_t[32];
    deviceMemcpy(output, weights, sizeof(uint64_t) * 32);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL64T32::SetWeights(uint64_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 32, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::SetWeights(uint64_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 32);
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T32::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::SetWeight(size_t Index, uint64_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL64T32::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint32_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::SetBias(uint32_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint32_t BrendanCUDA::AI::MLPB::MLPBL64T32::Run(uint64_t Input) const {
#if __CUDA_ARCH__
    uint32_t o = 0i32;
    if (Input & weights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & weights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & weights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & weights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & weights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & weights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & weights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & weights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & weights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & weights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & weights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & weights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & weights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & weights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & weights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & weights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & weights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & weights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & weights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & weights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & weights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & weights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & weights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & weights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & weights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & weights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & weights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint32_t* hWeights = new uint32_t[64];
    uint32_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint32_t) * 64, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & hWeights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & hWeights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & hWeights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & hWeights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & hWeights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & hWeights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & hWeights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & hWeights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & hWeights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & hWeights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & hWeights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & hWeights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & hWeights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & hWeights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & hWeights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & hWeights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & hWeights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & hWeights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & hWeights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & hWeights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T32::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint64_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::CopyTo(MLPBL64T32 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint64_t) * 32);
    deviceMemcpy(Other.bias, bias, sizeof(uint32_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint64_t) * 32, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T32 BrendanCUDA::AI::MLPB::MLPBL64T32::Clone() const {
    MLPBL64T32 n = MLPBL64T32();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T32 BrendanCUDA::AI::MLPB::MLPBL64T32::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T32 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T32 BrendanCUDA::AI::MLPB::MLPBL64T32::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T32 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T32::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint64_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint32_t v = (uint32_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint32_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T32 BrendanCUDA::AI::MLPB::MLPBL64T32::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T32 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL64T32::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL64T32 BrendanCUDA::AI::MLPB::MLPBL64T32::Deserialize(std::basic_istream<char>& Stream) {
//
//}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T64::MLPBL64T64() {
#if __CUDA_ARCH__
    weights = new uint64_t[64];
    bias = new uint64_t;
#else
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
#endif
}
__host__ BrendanCUDA::AI::MLPB::MLPBL64T64::MLPBL64T64(uint64_t* Weights, uint64_t* Bias, bool CopyFromHost) {
    ThrowIfBad(cudaMalloc(&weights, sizeof(uint64_t) * 64));
    ThrowIfBad(cudaMalloc(&bias, sizeof(uint64_t)));
    auto t = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 64, t));
    ThrowIfBad(cudaMemcpy(bias, Bias, sizeof(uint64_t), t));
}
__device__ BrendanCUDA::AI::MLPB::MLPBL64T64::MLPBL64T64(uint64_t* Weights, uint64_t* Bias) {
    weights = new uint64_t[64];
    bias = new uint64_t;
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 64);
    *bias = *Bias;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T64::MLPBL64T64(uint64_t* Weights, uint64_t Bias)
    : MLPBL64T64() {
#if __CUDA_ARCH__
    SetWeights(Weights);
#else
    SetWeights(Weights, true);
#endif
    SetBias(Bias);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::Dispose() {
#if __CUDA_ARCH__
    delete [] weights;
    delete [] bias;
#else
    ThrowIfBad(cudaFree(weights));
    ThrowIfBad(cudaFree(bias));
#endif
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T64::Weights() const {
    return weights;
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T64::Weight(size_t Index) const {
    return &weights[Index];
}
__host__ __device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T64::Bias() const {
    return bias;
}
__host__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T64::GetWeights(bool CopyToHost) const {
    if (CopyToHost) {
        uint64_t* output = new uint64_t[64];
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 64, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        uint64_t* output;
        ThrowIfBad(cudaMalloc(&output, sizeof(uint64_t) * 64));
        ThrowIfBad(cudaMemcpy(output, weights, sizeof(uint64_t) * 64, cudaMemcpyDeviceToDevice));
        return output;
    }
}
__device__ uint64_t* BrendanCUDA::AI::MLPB::MLPBL64T64::GetWeights() const {
    uint64_t* output = new uint64_t[64];
    deviceMemcpy(output, weights, sizeof(uint64_t) * 64);
    return output;
}
__host__ void BrendanCUDA::AI::MLPB::MLPBL64T64::SetWeights(uint64_t* Weights, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(weights, Weights, sizeof(uint64_t) * 64, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::SetWeights(uint64_t* Weights) {
    deviceMemcpy(weights, Weights, sizeof(uint64_t) * 64);
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T64::GetWeight(size_t Index) const {
#if __CUDA_ARCH__
    return weights[Index];
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, &weights[Index], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::SetWeight(size_t Index, uint64_t Weight) {
#if __CUDA_ARCH__
    weights[Index] = Weight;
#else
    ThrowIfBad(cudaMemcpy(&weights[Index], &Weight, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T64::GetBias() const {
#if __CUDA_ARCH__
    return *bias;
#else
    uint64_t output;
    ThrowIfBad(cudaMemcpy(&output, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return output;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::SetBias(uint64_t Bias) {
#if __CUDA_ARCH__
    *bias = Bias;
#else
    ThrowIfBad(cudaMemcpy(bias, &Bias, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T64::Run(uint64_t Input) const {
#if __CUDA_ARCH__
    uint64_t o = 0i64;
    if (Input & weights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & weights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & weights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & weights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & weights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & weights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & weights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & weights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & weights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & weights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & weights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & weights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & weights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & weights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & weights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & weights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & weights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & weights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & weights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & weights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & weights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & weights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & weights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & weights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & weights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & weights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & weights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & weights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & weights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & weights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & weights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & weights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & weights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & weights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & weights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & weights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & weights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & weights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & weights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & weights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & weights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & weights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & weights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & weights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & weights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & weights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & weights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & weights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & weights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & weights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & weights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & weights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & weights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & weights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & weights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & weights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & weights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & weights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & weights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o ^ (*bias);
#else
    uint64_t* hWeights = new uint64_t[64];
    uint64_t o;
    ThrowIfBad(cudaMemcpy(hWeights, weights, sizeof(uint64_t) * 64, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(&o, bias, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (Input & hWeights[0])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000001;
    if (Input & hWeights[1])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000010;
    if (Input & hWeights[2])
        o |= 0b0000000000000000000000000000000000000000000000000000000000000100;
    if (Input & hWeights[3])
        o |= 0b0000000000000000000000000000000000000000000000000000000000001000;
    if (Input & hWeights[4])
        o |= 0b0000000000000000000000000000000000000000000000000000000000010000;
    if (Input & hWeights[5])
        o |= 0b0000000000000000000000000000000000000000000000000000000000100000;
    if (Input & hWeights[6])
        o |= 0b0000000000000000000000000000000000000000000000000000000001000000;
    if (Input & hWeights[7])
        o |= 0b0000000000000000000000000000000000000000000000000000000010000000;
    if (Input & hWeights[8])
        o |= 0b0000000000000000000000000000000000000000000000000000000100000000;
    if (Input & hWeights[9])
        o |= 0b0000000000000000000000000000000000000000000000000000001000000000;
    if (Input & hWeights[10])
        o |= 0b0000000000000000000000000000000000000000000000000000010000000000;
    if (Input & hWeights[11])
        o |= 0b0000000000000000000000000000000000000000000000000000100000000000;
    if (Input & hWeights[12])
        o |= 0b0000000000000000000000000000000000000000000000000001000000000000;
    if (Input & hWeights[13])
        o |= 0b0000000000000000000000000000000000000000000000000010000000000000;
    if (Input & hWeights[14])
        o |= 0b0000000000000000000000000000000000000000000000000100000000000000;
    if (Input & hWeights[15])
        o |= 0b0000000000000000000000000000000000000000000000001000000000000000;
    if (Input & hWeights[16])
        o |= 0b0000000000000000000000000000000000000000000000010000000000000000;
    if (Input & hWeights[17])
        o |= 0b0000000000000000000000000000000000000000000000100000000000000000;
    if (Input & hWeights[18])
        o |= 0b0000000000000000000000000000000000000000000001000000000000000000;
    if (Input & hWeights[19])
        o |= 0b0000000000000000000000000000000000000000000010000000000000000000;
    if (Input & hWeights[20])
        o |= 0b0000000000000000000000000000000000000000000100000000000000000000;
    if (Input & hWeights[21])
        o |= 0b0000000000000000000000000000000000000000001000000000000000000000;
    if (Input & hWeights[22])
        o |= 0b0000000000000000000000000000000000000000010000000000000000000000;
    if (Input & hWeights[23])
        o |= 0b0000000000000000000000000000000000000000100000000000000000000000;
    if (Input & hWeights[24])
        o |= 0b0000000000000000000000000000000000000001000000000000000000000000;
    if (Input & hWeights[25])
        o |= 0b0000000000000000000000000000000000000010000000000000000000000000;
    if (Input & hWeights[26])
        o |= 0b0000000000000000000000000000000000000100000000000000000000000000;
    if (Input & hWeights[27])
        o |= 0b0000000000000000000000000000000000001000000000000000000000000000;
    if (Input & hWeights[28])
        o |= 0b0000000000000000000000000000000000010000000000000000000000000000;
    if (Input & hWeights[29])
        o |= 0b0000000000000000000000000000000000100000000000000000000000000000;
    if (Input & hWeights[30])
        o |= 0b0000000000000000000000000000000001000000000000000000000000000000;
    if (Input & hWeights[31])
        o |= 0b0000000000000000000000000000000010000000000000000000000000000000;
    if (Input & hWeights[32])
        o |= 0b0000000000000000000000000000000100000000000000000000000000000000;
    if (Input & hWeights[33])
        o |= 0b0000000000000000000000000000001000000000000000000000000000000000;
    if (Input & hWeights[34])
        o |= 0b0000000000000000000000000000010000000000000000000000000000000000;
    if (Input & hWeights[35])
        o |= 0b0000000000000000000000000000100000000000000000000000000000000000;
    if (Input & hWeights[36])
        o |= 0b0000000000000000000000000001000000000000000000000000000000000000;
    if (Input & hWeights[37])
        o |= 0b0000000000000000000000000010000000000000000000000000000000000000;
    if (Input & hWeights[38])
        o |= 0b0000000000000000000000000100000000000000000000000000000000000000;
    if (Input & hWeights[39])
        o |= 0b0000000000000000000000001000000000000000000000000000000000000000;
    if (Input & hWeights[40])
        o |= 0b0000000000000000000000010000000000000000000000000000000000000000;
    if (Input & hWeights[41])
        o |= 0b0000000000000000000000100000000000000000000000000000000000000000;
    if (Input & hWeights[42])
        o |= 0b0000000000000000000001000000000000000000000000000000000000000000;
    if (Input & hWeights[43])
        o |= 0b0000000000000000000010000000000000000000000000000000000000000000;
    if (Input & hWeights[44])
        o |= 0b0000000000000000000100000000000000000000000000000000000000000000;
    if (Input & hWeights[45])
        o |= 0b0000000000000000001000000000000000000000000000000000000000000000;
    if (Input & hWeights[46])
        o |= 0b0000000000000000010000000000000000000000000000000000000000000000;
    if (Input & hWeights[47])
        o |= 0b0000000000000000100000000000000000000000000000000000000000000000;
    if (Input & hWeights[48])
        o |= 0b0000000000000001000000000000000000000000000000000000000000000000;
    if (Input & hWeights[49])
        o |= 0b0000000000000010000000000000000000000000000000000000000000000000;
    if (Input & hWeights[50])
        o |= 0b0000000000000100000000000000000000000000000000000000000000000000;
    if (Input & hWeights[51])
        o |= 0b0000000000001000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[52])
        o |= 0b0000000000010000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[53])
        o |= 0b0000000000100000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[54])
        o |= 0b0000000001000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[55])
        o |= 0b0000000010000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[56])
        o |= 0b0000000100000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[57])
        o |= 0b0000001000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[58])
        o |= 0b0000010000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[59])
        o |= 0b0000100000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[60])
        o |= 0b0001000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[61])
        o |= 0b0010000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[62])
        o |= 0b0100000000000000000000000000000000000000000000000000000000000000;
    if (Input & hWeights[63])
        o |= 0b1000000000000000000000000000000000000000000000000000000000000000;
    return o;
#endif
}
__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBL64T64::RunG(uint64_t Input) const {
    return (uint64_t)Run((uint64_t)Input);
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::CopyTo(MLPBL64T64 Other) const {
#if __CUDA_ARCH__
    deviceMemcpy(Other.weights, weights, sizeof(uint64_t) * 64);
    deviceMemcpy(Other.bias, bias, sizeof(uint64_t));
#else
    ThrowIfBad(cudaMemcpy(Other.weights, weights, sizeof(uint64_t) * 64, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(Other.bias, bias, sizeof(uint64_t), cudaMemcpyDeviceToDevice));
#endif
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T64 BrendanCUDA::AI::MLPB::MLPBL64T64::Clone() const {
    MLPBL64T64 n = MLPBL64T64();
    CopyTo(n);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(weights, 64, WeightsFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T64 BrendanCUDA::AI::MLPB::MLPBL64T64::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T64 n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    applyTargetFlipsOnArray(weights, 64, WeightsEachFlipProb, rng);
    RandomizeArray(bias, 1, BiasFlipProb, rng);
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T64 BrendanCUDA::AI::MLPB::MLPBL64T64::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T64 n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBL64T64::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (uint32_t i = 0; i < 64; ++i) {
        if (rng() < WeightsMutationProb) {
            uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(WeightsProbOf1, rng);

#if __CUDA_ARCH__
            deviceMemcpy(weights + i, &v, sizeof(uint64_t));
#else
            ThrowIfBad(cudaMemcpy(weights + i, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
        }
    }
    if (rng() < WeightsMutationProb) {
        uint64_t v = (uint64_t)BrendanCUDA::Random::Get64Bits(BiasProbOf1, rng);
        
#if __CUDA_ARCH__
        deviceMemcpy(bias, &v, sizeof(uint64_t));
#else
        ThrowIfBad(cudaMemcpy(bias, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL64T64 BrendanCUDA::AI::MLPB::MLPBL64T64::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPBL64T64 n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}
//__host__ void BrendanCUDA::AI::MLPB::MLPBL64T64::Serialize(std::basic_ostream<char>& Stream) {
//
//}
//__host__ BrendanCUDA::AI::MLPB::MLPBL64T64 BrendanCUDA::AI::MLPB::MLPBL64T64::Deserialize(std::basic_istream<char>& Stream) {
//
//}
