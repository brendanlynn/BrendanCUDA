#include "brendancuda_ai_mlpb.cuh"

__host__ __device__ BrendanCUDA::AI::MLPB::MLPB::MLPB(size_t LayerCount) {
#if IS_ON_DEVICE
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
#if IS_ON_HOST
        GetLayer(i).Dispose();
#else
        layers[i].Dispose();
#endif
    }
#if IS_ON_HOST
    cudaFree(layers);
#else
    delete[] layers;
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
#if IS_ON_DEVICE
    return layers[LayerIndex];
#else
    MLPBL r;
#if _DEBUG
    auto tm1 = cudaDeviceSynchronize();
    auto t0 =
#endif
        cudaMemcpy(&r, &layers[LayerIndex], sizeof(MLPBL), cudaMemcpyDeviceToHost);
    return r;
#endif
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::SetLayer(size_t LayerIndex, MLPBL Layer) {
#if IS_ON_DEVICE
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
#if IS_ON_HOST
    cudaMemcpy(r.layers, layers, d, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&r.layers[layerCount], layers, d, cudaMemcpyDeviceToDevice);
#else
    deviceMemcpy(r.layers, layers, d);
    deviceMemcpy(&r.layers[layerCount], layers, d);
#endif
    return r;
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::operator+(const MLPBL Value) {
    MLPB r = MLPB(layerCount + 1);
#if IS_ON_HOST
    cudaMemcpy(r.layers, layers, sizeof(MLPBL) * layerCount, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&r.layers[layerCount], &Value, sizeof(MLPBL), cudaMemcpyDeviceToDevice);
#else
    deviceMemcpy(r.layers, layers, sizeof(MLPBL) * layerCount);
    deviceMemcpy(&r.layers[layerCount], &Value, sizeof(MLPBL));
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
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) {
    for (size_t i = 0; i < layerCount; ++i) {
#if IS_ON_HOST
        GetLayer(i).RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
#else
        Layer(i)->RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWFlips(uint32_t WeightsFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const {
    MLPB n = Clone();
    n.RandomizeWFlips(WeightsFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) {
    for (size_t i = 0; i < layerCount; ++i) {
#if IS_ON_HOST
        GetLayer(i).RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
#else
        Layer(i)->RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWTargets(uint32_t WeightsEachFlipProb, uint32_t BiasFlipProb, Random::rngWState<uint64_t> rng) const {
    MLPB n = Clone();
    n.RandomizeWTargets(WeightsEachFlipProb, BiasFlipProb, rng);
    return n;
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPB::RandomizeWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) {
    for (size_t i = 0; i < layerCount; ++i) {
#if IS_ON_HOST
        GetLayer(i).RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
#else
        Layer(i)->RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
#endif
    }
}
__host__ __device__ BrendanCUDA::AI::MLPB::MLPB BrendanCUDA::AI::MLPB::MLPB::ReproduceWMutations(uint32_t WeightsMutationProb, uint32_t WeightsProbOf1, uint32_t BiasMutationProb, uint32_t BiasProbOf1, Random::rngWState<uint64_t> rng) const {
    MLPB n = Clone();
    n.RandomizeWMutations(WeightsMutationProb, WeightsProbOf1, BiasMutationProb, BiasProbOf1, rng);
    return n;
}