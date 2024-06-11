#include "brendancuda_ai_mlpb.cuh"

__host__ __device__ BrendanCUDA::AI::MLPB::MLPB::MLPB(size_t LayerCount) {
#if __CUDA_ARCH__
    layers = new MLPBL[LayerCount];
#else
    cudaMalloc(&layers, sizeof(MLPBL) * LayerCount);
#endif
    layerCount = LayerCount;
}
__host__ BrendanCUDA::AI::MLPB::MLPB::MLPB(MLPBL* Layers, size_t LayerCount, bool CopyFromHost)
    : MLPB(LayerCount) {
    cudaMemcpy(layers, Layers, sizeof(MLPBL) * LayerCount, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
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
    cudaFree(layers);
#endif
}

__host__ BrendanCUDA::AI::MLPB::MLPBL* BrendanCUDA::AI::MLPB::MLPB::GetLayers(bool CopyToHost) const {
    MLPBL* p;
    if (CopyToHost) {
        p = new MLPBL[layerCount];
        cudaMemcpy(p, layers, sizeof(MLPBL) * layerCount, cudaMemcpyDeviceToHost);
    }
    else {
        cudaMalloc(&p, sizeof(MLPBL) * layerCount);
        cudaMemcpy(p, layers, sizeof(MLPBL) * layerCount, cudaMemcpyDeviceToDevice);
    }
    return p;
}
__device__ BrendanCUDA::AI::MLPB::MLPBL* BrendanCUDA::AI::MLPB::MLPB::GetLayers() const {
    MLPBL* p = new MLPBL[layerCount];
    deviceMemcpy(p, layers, sizeof(MLPBL) * layerCount);
    return p;
}
__host__ void BrendanCUDA::AI::MLPB::MLPB::SetLayers(MLPBL* Layers, bool CopyFromHost) {
    cudaMemcpy(layers, Layers, sizeof(MLPBL) * layerCount, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
}
__device__ void BrendanCUDA::AI::MLPB::MLPB::SetLayers(MLPBL* Layers) {
    deviceMemcpy(layers, Layers, sizeof(MLPBL) * layerCount);
}

__host__ __device__ BrendanCUDA::AI::MLPB::MLPBL BrendanCUDA::AI::MLPB::MLPB::GetLayer(size_t LayerIndex) const {
#if __CUDA_ARCH__
    return layers[LayerIndex];
#else
    MLPBL r;
    auto e = cudaMemcpy(&r, &layers[LayerIndex], sizeof(MLPBL), cudaMemcpyDeviceToHost);
    if (e) {
        throw std::exception();
    }
    return r;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::SetLayer(size_t LayerIndex, MLPBL Layer) {
#if __CUDA_ARCH__
    layers[LayerIndex] = Layer;
#else
    cudaMemcpy(&layers[LayerIndex], &Layer, sizeof(MLPBL), cudaMemcpyHostToDevice);
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
    cudaMemcpy(r.layers, layers, d, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&r.layers[layerCount], layers, d, cudaMemcpyDeviceToDevice);
#endif
    return r;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::operator+(const MLPBL Value) {
    MLPB r = MLPB(layerCount + 1);
#if __CUDA_ARCH__
    deviceMemcpy(r.layers, layers, sizeof(MLPBL) * layerCount);
    deviceMemcpy(&r.layers[layerCount], &Value, sizeof(MLPBL));
#else
    cudaMemcpy(r.layers, layers, sizeof(MLPBL) * layerCount, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&r.layers[layerCount], &Value, sizeof(MLPBL), cudaMemcpyDeviceToDevice);
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
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < layerCount; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
#else
        GetLayer(i).RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPB n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < layerCount; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
#else
        GetLayer(i).RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::AnyRNG<uint64_t> rng) const {
    MLPB n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < layerCount; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
#else
        GetLayer(i).RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::AnyRNG<uint64_t> rng) const {
    MLPB n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}