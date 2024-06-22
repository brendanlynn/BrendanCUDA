#include "brendancuda_ai_mlpb.cuh"
#include "brendancuda_cudaerrorhelpers.h"

__host__ __device__ BrendanCUDA::AI::MLPB::MLPB::MLPB(size_t LayerCount) {
#if __CUDA_ARCH__
    layers = new MLPBL[LayerCount];
#else
    ThrowIfBad(cudaMalloc(&layers, sizeof(MLPBL) * LayerCount));
#endif
    layerCount = LayerCount;
}
__host__ BrendanCUDA::AI::MLPB::MLPB::MLPB(MLPBL* Layers, size_t LayerCount, bool CopyFromHost)
    : MLPB(LayerCount) {
    ThrowIfBad(cudaMemcpy(layers, Layers, sizeof(MLPBL) * LayerCount, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ BrendanCUDA::AI::MLPB::MLPB::MLPB(MLPBL* Layers, size_t LayerCount) {
    layers = Layers;
    layerCount = LayerCount;
}

__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::Dispose() {
    for (size_t i = 0; i < layerCount; ++i) {
#if __CUDA_ARCH__
        layers[i].Dispose();
#else
        GetLayer(i).Dispose();
#endif
    }
#if __CUDA_ARCH__
    delete[] layers;
#else
    ThrowIfBad(cudaFree(layers));
#endif
}

__host__ BrendanCUDA::AI::MLPB::MLPBL* BrendanCUDA::AI::MLPB::MLPB::GetLayers(bool CopyToHost) const {
    MLPBL* p;
    if (CopyToHost) {
        p = new MLPBL[layerCount];
        ThrowIfBad(cudaMemcpy(p, layers, sizeof(MLPBL) * layerCount, cudaMemcpyDeviceToHost));
    }
    else {
        ThrowIfBad(cudaMalloc(&p, sizeof(MLPBL) * layerCount));
        ThrowIfBad(cudaMemcpy(p, layers, sizeof(MLPBL) * layerCount, cudaMemcpyDeviceToDevice));
    }
    return p;
}
__device__ BrendanCUDA::AI::MLPB::MLPBL* BrendanCUDA::AI::MLPB::MLPB::GetLayers() const {
    MLPBL* p = new MLPBL[layerCount];
    deviceMemcpy(p, layers, sizeof(MLPBL) * layerCount);
    return p;
}
__host__ void BrendanCUDA::AI::MLPB::MLPB::SetLayers(MLPBL* Layers, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(layers, Layers, sizeof(MLPBL) * layerCount, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
__device__ void BrendanCUDA::AI::MLPB::MLPB::SetLayers(MLPBL* Layers) {
    deviceMemcpy(layers, Layers, sizeof(MLPBL) * layerCount);
}

__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL BrendanCUDA::AI::MLPB::MLPB::GetLayer(size_t LayerIndex) const {
#if __CUDA_ARCH__
    return layers[LayerIndex];
#else
    MLPBL r;
    ThrowIfBad(cudaMemcpy(&r, &layers[LayerIndex], sizeof(MLPBL), cudaMemcpyDeviceToHost));
    return r;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::SetLayer(size_t LayerIndex, MLPBL Layer) {
#if __CUDA_ARCH__
    layers[LayerIndex] = Layer;
#else
    ThrowIfBad(cudaMemcpy(&layers[LayerIndex], &Layer, sizeof(MLPBL), cudaMemcpyHostToDevice));
#endif
}

__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL* BrendanCUDA::AI::MLPB::MLPB::Layers() const {
    return layers;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL* BrendanCUDA::AI::MLPB::MLPB::Layer(size_t LayerIndex) const {
    return &layers[LayerIndex];
}
__host__ __device__ size_t BrendanCUDA::AI::MLPB::MLPB::LayerCount() const {
    return layerCount;
}

__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPB::Run(uint64_t Input) const {
    if (layerCount == 0)
        return 0;
    for (size_t i = 0; i < layerCount; ++i) {
        Input = layers[i].Run(Input);
    }
    return Input;
}

__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::operator+(const MLPB Value) {
    MLPB r = MLPB(layerCount + Value.layerCount);
    size_t d = sizeof(MLPBL) * layerCount;
#if __CUDA_ARCH__
    deviceMemcpy(r.layers, layers, d);
    deviceMemcpy(&r.layers[layerCount], layers, d);
#else
    ThrowIfBad(cudaMemcpy(r.layers, layers, d, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(&r.layers[layerCount], layers, d, cudaMemcpyDeviceToDevice));
#endif
    return r;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::operator+(const MLPBL Value) {
    MLPB r = MLPB(layerCount + 1);
#if __CUDA_ARCH__
    deviceMemcpy(r.layers, layers, sizeof(MLPBL) * layerCount);
    deviceMemcpy(&r.layers[layerCount], &Value, sizeof(MLPBL));
#else
    ThrowIfBad(cudaMemcpy(r.layers, layers, sizeof(MLPBL) * layerCount, cudaMemcpyDeviceToDevice));
    ThrowIfBad(cudaMemcpy(&r.layers[layerCount], &Value, sizeof(MLPBL), cudaMemcpyDeviceToDevice));
#endif
    return r;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::Clone() const {
    MLPB n(layerCount);
    for (size_t i = 0; i < layerCount; ++i) {
        n.SetLayer(i, GetLayer(i).Clone());
    }
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < layerCount; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
#else
        GetLayer(i).RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    MLPB n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, RNG);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < layerCount; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
#else
        GetLayer(i).RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> RNG) const {
    MLPB n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, RNG);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) {
    for (size_t i = 0; i < layerCount; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
#else
        GetLayer(i).RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> RNG) const {
    MLPB n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, RNG);
    return n;
}