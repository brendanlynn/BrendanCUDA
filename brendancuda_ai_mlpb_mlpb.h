#pragma once

#include "brendancuda_ai_mlpb_mlpbl.h"
#include "brendancuda_ai_mlpb_mlpblw.h"
#include "brendancuda_dcopy.cuh"
#include "brendancuda_errorhelp.h"
#include "brendancuda_rand_anyrng.h"
#include "BSerializer/Serializer.h"
#include <memory>

namespace BrendanCUDA {
    namespace AI {
        namespace MLPB {
            class [[deprecated("out of support")]] MLPB {
            public:
                __host__ __device__ __forceinline MLPB(size_t LayerCount);
                __host__ __device__ __forceinline MLPB(MLPBLW* Layers, size_t LayerCount);

                __host__ __device__ __forceinline void Dispose();

                __host__ __device__ __forceinline MLPBLW* Layers() const;
                __host__ __device__ __forceinline MLPBLW* Layer(size_t Index) const;
                __host__ __device__ __forceinline size_t LayerCount() const;

                template <bool _CopyToHost>
                __host__ __forceinline MLPBLW* CopyOutLayers() const;
                __host__ __device__ __forceinline void CopyOutLayers(MLPBLW* Layers) const;
#ifdef __CUDACC__
                __device__ __forceinline MLPBLW* CopyOutLayers() const;
#endif
                __host__ __device__ __forceinline void CopyInLayers(MLPBLW* Layers);

                __host__ __device__ __forceinline MLPBLW GetLayer(size_t Index) const;
                __host__ __device__ __forceinline void SetLayer(size_t Index, MLPBLW Layer);

                __host__ __device__ __forceinline uint64_t Run(uint64_t Input) const;

                __host__ __device__ __forceinline MLPB operator+(const MLPB Value) const;
                __host__ __device__ __forceinline MLPB operator+(const MLPBLW Value) const;

                __host__ __device__ __forceinline MLPB Clone() const;
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline void RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline MLPB ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const;
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline void RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline MLPB ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const;
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline void RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                __host__ __device__ __forceinline MLPB ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) const;

                __forceinline size_t SerializedSize() const;
                __forceinline void Serialize(void*& Data) const;
                static __forceinline MLPB Deserialize(const void*& Data);
                static __forceinline void Deserialize(const void*& Data, void* Value);
            private:
                MLPBLW* layers;
                size_t layerCount;
            };
        }
    }
}

__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPB::MLPB(size_t LayerCount) {
#ifdef __CUDA_ARCH__
    layers = new MLPBLW[LayerCount];
#else
    ThrowIfBad(cudaMalloc(&layers, sizeof(MLPBLW) * LayerCount));
#endif
    layerCount = LayerCount;
}
__host__ __device__ __forceinline BrendanCUDA::AI::MLPB::MLPB::MLPB(MLPBLW* Layers, size_t LayerCount) {
    layers = Layers;
    layerCount = LayerCount;
}

__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPB::Dispose() {
    for (size_t i = 0; i < layerCount; ++i) {
#ifdef __CUDA_ARCH__
        layers[i].Dispose();
#else
        GetLayer(i).Dispose();
#endif
    }
#ifdef __CUDA_ARCH__
    delete[] layers;
#else
    ThrowIfBad(cudaFree(layers));
#endif
}

template <bool _CopyToHost>
__host__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::CopyOutLayers() const -> MLPBLW* {
    MLPBLW* p;
    if constexpr (_CopyToHost) {
        p = new MLPBLW[layerCount];
        ThrowIfBad(cudaMemcpy(p, layers, sizeof(MLPBLW) * layerCount, cudaMemcpyDeviceToHost));
    }
    else {
        ThrowIfBad(cudaMalloc(&p, sizeof(MLPBLW) * layerCount));
        ThrowIfBad(cudaMemcpy(p, layers, sizeof(MLPBLW) * layerCount, cudaMemcpyDeviceToDevice));
    }
    return p;
}
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPB::CopyOutLayers(MLPBLW* Layers) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Layers, layers, sizeof(MLPBLW) * layerCount);
#else
    ThrowIfBad(cudaMemcpy(Layers, layers, sizeof(MLPBLW) * layerCount, cudaMemcpyDefault));
#endif
}
#ifdef __CUDACC__
__device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::CopyOutLayers() const -> MLPBLW* {
    MLPBLW* p = new MLPBLW[layerCount];
    deviceMemcpy(p, layers, sizeof(MLPBLW) * layerCount);
    return p;
}
#endif
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPB::CopyInLayers(MLPBLW* Layers) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(layers, Layers, sizeof(MLPBLW) * layerCount);
#else
    ThrowIfBad(cudaMemcpy(layers, Layers, sizeof(MLPBLW) * layerCount, cudaMemcpyDefault));
#endif
}

__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::GetLayer(size_t Index) const -> MLPBLW {
#ifdef __CUDA_ARCH__
    return layers[Index];
#else
    MLPBLW r;
    ThrowIfBad(cudaMemcpy(&r, layers + Index, sizeof(MLPBLW), cudaMemcpyDeviceToHost));
    return r;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::SetLayer(size_t Index, MLPBLW Layer) {
#ifdef __CUDA_ARCH__
    layers[Index] = Layer;
#else
    ThrowIfBad(cudaMemcpy(layers + Index, &Layer, sizeof(MLPBLW), cudaMemcpyHostToDevice));
#endif
}

__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::Layers() const -> MLPBLW* {
    return layers;
}
__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::Layer(size_t Index) const -> MLPBLW* {
    return layers + Index;
}
__host__ __device__ __forceinline size_t BrendanCUDA::AI::MLPB::MLPB::LayerCount() const {
    return layerCount;
}

__host__ __device__ __forceinline uint64_t BrendanCUDA::AI::MLPB::MLPB::Run(uint64_t Input) const {
    if (layerCount == 0) return 0;
    for (size_t i = 0; i < layerCount; ++i) {
        Input = layers[i].Run(Input);
    }
    return Input;
}

__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::operator+(const MLPB Value) const -> MLPB {
    MLPB r = MLPB(layerCount + Value.layerCount);
    MLPB a = Clone();
    MLPB b = Value.Clone();
#ifdef __CUDA_ARCH__
    deviceMemcpy(r.layers, a.layers, sizeof(MLPBLW) * layerCount);
    deviceMemcpy(r.layers + layerCount, b.layers, sizeof(MLPBLW) * Value.layerCount);
#else
    ThrowIfBad(cudaMemcpy(r.layers, a.layers, sizeof(MLPBLW) * layerCount, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(r.layers + layerCount, b.layers, sizeof(MLPBLW) * Value.layerCount, cudaMemcpyDeviceToDevice));
#endif
    return r;
}
__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::operator+(const MLPBLW Value) const -> MLPB {
    MLPB r = MLPB(layerCount + 1);
    MLPB a = Clone();
    MLPBLW b = Value.Clone();
#ifdef __CUDA_ARCH__
    deviceMemcpy(r.layers, a.layers, sizeof(MLPBLW) * layerCount);
    deviceMemcpy(r.layers + layerCount, &b, sizeof(MLPBLW));
#else
    ThrowIfBad(cudaMemcpy(r.layers, a.layers, sizeof(MLPBLW) * layerCount, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(r.layers + layerCount, &b, sizeof(MLPBLW), cudaMemcpyDeviceToDevice));
#endif
    return r;
}
__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::Clone() const -> MLPB {
    MLPB n(layerCount);
    for (size_t i = 0; i < layerCount; ++i) {
        n.SetLayer(i, GetLayer(i).Clone());
    }
    return n;
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPB::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    for (size_t i = 0; i < layerCount; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
#else
        GetLayer(i).RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
#endif
    }
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const -> MLPB {
    MLPB n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
    return n;
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPB::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) {
    for (size_t i = 0; i < layerCount; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
#else
        GetLayer(i).RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
#endif
    }
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, _TRNG& RNG) const -> MLPB {
    MLPB n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
    return n;
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline void BrendanCUDA::AI::MLPB::MLPB::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) {
    for (size_t i = 0; i < layerCount; ++i) {
#ifdef __CUDA_ARCH__
        Layer(i)->RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
#else
        GetLayer(i).RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
#endif
    }
}
template <std::uniform_random_bit_generator _TRNG>
__host__ __device__ __forceinline auto BrendanCUDA::AI::MLPB::MLPB::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, _TRNG& RNG) const -> MLPB {
    MLPB n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
    return n;
}

__forceinline size_t BrendanCUDA::AI::MLPB::MLPB::SerializedSize() const {
    size_t t = sizeof(size_t);
    for (size_t i = 0; i < layerCount; ++i) {
        t += GetLayer(i).SerializedSize();
    }
    return t;
}
__forceinline void BrendanCUDA::AI::MLPB::MLPB::Serialize(void*& Data) const {
    BSerializer::Serialize(Data, layerCount);
    for (size_t i = 0; i < layerCount; ++i) {
        GetLayer(i).Serialize(Data);
    }
}
__forceinline auto BrendanCUDA::AI::MLPB::MLPB::Deserialize(const void*& Data) -> MLPB {
    MLPB mlpb(BSerializer::Deserialize<size_t>(Data));
    for (size_t i = 0; i < mlpb.layerCount; ++i) {
        mlpb.SetLayer(i, MLPBLW::Deserialize(Data));
    }
    return mlpb;
}
__forceinline void BrendanCUDA::AI::MLPB::MLPB::Deserialize(const void*& Data, void* ObjMem) {
    new (ObjMem) MLPB(BSerializer::Deserialize<size_t>(Data));
    MLPB& obj = *(MLPB*)ObjMem;
    for (size_t i = 0; i < obj.layerCount; ++i) {
        obj.SetLayer(i, MLPBLW::Deserialize(Data));
    }
}