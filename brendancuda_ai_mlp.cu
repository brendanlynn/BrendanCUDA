#include "brendancuda_ai_mlp.cuh"

template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<T>::MLP(size_t Length, activationFunction_t<T> ActivationFunction) {
    len = Length;
    actnFunc = ActivationFunction;
#if __CUDA_ARCH__
    lyrs = new MLPL<T>[Length];
#else
    cudaMalloc(&lyrs, Length * sizeof(MLPL<T>));
#endif
}
template <typename T>
__host__ BrendanCUDA::AI::MLP::MLP<T>::MLP(size_t Length, activationFunction_t<T> ActivationFunction, MLPL<T>* Layers, bool CopyFromHost)
    : MLP(Length, ActivationFunction) {
    cudaMemcpy(lyrs, Layers, Length * sizeof(MLPL<T>), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
}
template <typename T>
__device__ BrendanCUDA::AI::MLP::MLP<T>::MLP(size_t Length, activationFunction_t<T> ActivationFunction, MLPL<T>* Layers)
    : MLP(Length, ActivationFunction) {
    deviceMemcpy(lyrs, Layers, Length * sizeof(MLPL<T>));
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::Dispose() {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->Dispose();
#else
        GetLayer(i).Dispose();
#endif
    }
#if __CUDA_ARCH__
    delete[] lyrs;
#else
    cudaFree(lyrs);
#endif
}
template <typename T>
__host__ std::pair<T*, size_t> BrendanCUDA::AI::MLP::MLP<T>::Run(T* Input) {
    if (len == 0) {
        return std::pair<T*, size_t>(0, 0);
    }
#if __CUDA_ARCH__
    MLPL<T>* l = Layer(0);
    Input = l->Run(Input);
    RunActivationFunctionOnArray(Input, l->OutputLength());
#else
    MLPL<T> l = GetLayer(0);
    Input = l.Run(Input);
#ifdef _DEBUG
    auto s0 = cudaDeviceSynchronize();
#endif
    RunActivationFunctionOnArray(Input, l.OutputLength());
#ifdef _DEBUG
    auto s1 = cudaDeviceSynchronize();
#endif
#endif
    for (size_t i = 1; i < len; ++i) {
#if __CUDA_ARCH__
        l = Layer(i);
        T* nxt = l->Run(Input);
        RunActivationFunctionOnArray(nxt, l->OutputLength());
        delete[] Input;
#else
        l = GetLayer(i);
        T* nxt = l.Run(Input);
        RunActivationFunctionOnArray(nxt, l.OutputLength());
        cudaFree(Input);
#endif
        Input = nxt;
    }
    return std::pair<T*, size_t>(Input, OutputLength());
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<T>* BrendanCUDA::AI::MLP::MLP<T>::Layers() const {
    return lyrs;
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<T>* BrendanCUDA::AI::MLP::MLP<T>::Layer(size_t LayerIndex) const {
    return &lyrs[LayerIndex];
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLP<T>::LayerCount() const {
    return len;
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::activationFunction_t<T> BrendanCUDA::AI::MLP::MLP<T>::ActivationFunction() const {
    return actnFunc;
}
template <typename T>
__host__ BrendanCUDA::AI::MLP::MLPL<T>* BrendanCUDA::AI::MLP::MLP<T>::GetLayers(bool CopyToHost) const {
    MLPL<T>* p;
    if (CopyToHost) {
        p = new MLPL<T>[len];
        cudaMemcpy(p, lyrs, sizeof(MLPL<T>) * len, cudaMemcpyDeviceToHost);
    }
    else {
        cudaMalloc(&p, sizeof(MLPL<T>) * len);
        cudaMemcpy(p, lyrs, sizeof(MLPL<T>) * len, cudaMemcpyDeviceToDevice);
    }
    return p;
}
template <typename T>
__device__ BrendanCUDA::AI::MLP::MLPL<T>* BrendanCUDA::AI::MLP::MLP<T>::GetLayers() const {
    MLPL<T>* p = new MLPL<T>[len];
    deviceMemcpy(p, lyrs, sizeof(MLPL<T>) * len);
    return p;
}
template <typename T>
__host__ void BrendanCUDA::AI::MLP::MLP<T>::SetLayers(MLPL<T>* Layers, bool CopyFromHost) {
    cudaMemcpy(lyrs, Layers, sizeof(MLPL<T>) * len, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
}
template <typename T>
__device__ void BrendanCUDA::AI::MLP::MLP<T>::SetLayers(MLPL<T>* Layers) {
    deviceMemcpy(lyrs, Layers, sizeof(MLPL<T>) * len);
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<T> BrendanCUDA::AI::MLP::MLP<T>::GetLayer(size_t LayerIndex) const {
#if __CUDA_ARCH__
    return lyrs[LayerIndex];
#else
    MLPL<T> r;
#ifdef _DEBUG
    auto x =
#endif
        cudaMemcpy(&r, &lyrs[LayerIndex], sizeof(MLPL<T>), cudaMemcpyDeviceToHost);
    return r;
#endif
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::SetLayer(size_t LayerIndex, MLPL<T> Layer) {
#if __CUDA_ARCH__
    lyrs[LayerIndex] = Layer;
#else
    cudaMemcpy(&lyrs[LayerIndex], &Layer, sizeof(MLPL<T>), cudaMemcpyHostToDevice);
#endif
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLP<T>::InputLength() {
    if (len == 0) {
        return 0;
    }
#if __CUDA_ARCH__
    return Layer(0)->InputLength();
#else
    return GetLayer(0).InputLength();
#endif
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLP<T>::OutputLength() {
    if (len == 0) {
        return 0;
    }
#if __CUDA_ARCH__
    return Layer(len - 1)->OutputLength();
#else
    return GetLayer(len - 1).OutputLength();
#endif
}
//template <typename T>
//__global__ void RunActivationFunctionOnArrayKernel(T* Array, BrendanCUDA::AI::activationFunction_t<T> ActivationFunction) {
//    /*T& p(Array[blockIdx.x]);
//    p = ActivationFunction(p);*/
//    Array[blockIdx.x] = ActivationFunction(Array[blockIdx.x]);
//}
__global__ void RunActivationFunctionOnArrayKernel(float* Array, BrendanCUDA::AI::activationFunction_t<float> ActivationFunction) {
    float& p(Array[blockIdx.x]);
    p = ActivationFunction(p);
    //p = p * 2.f;
}
__global__ void RunActivationFunctionOnArrayKernel(double* Array, BrendanCUDA::AI::activationFunction_t<double> ActivationFunction) {
    double& p(Array[blockIdx.x]);
    p = ActivationFunction(p);
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::RunActivationFunctionOnArray(T* Array, size_t Length) {
    activationFunction_t<T> af = actnFunc;
#if __CUDA_ARCH__
    for (size_t i = 0; i < Length; ++i) {
        T& p(Array[i]);
        p = af(p);
    }
#else
    RunActivationFunctionOnArrayKernel<<<Length, 1 >>>(Array, af);
#endif
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<T> BrendanCUDA::AI::MLP::MLP<T>::Clone() const {
    MLP<T> m(len, actnFunc);
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        m.SetLayer(i, Layer(i)->Clone());
#else
        m.SetLayer(i, GetLayer(i).Clone());
#endif
    }
    return m;
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::Randomize(T Scalar, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->Randomize(Scalar, rng);
#else
        GetLayer(i).Randomize(Scalar, rng);
#endif
    }
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::Randomize(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->Randomize(Scalar, LowerBound, UpperBound, rng);
#else
        GetLayer(i).Randomize(Scalar, LowerBound, UpperBound, rng);
#endif
    }
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<T> BrendanCUDA::AI::MLP::MLP<T>::Reproduce(T Scalar, Random::AnyRNG<uint64_t> rng) const {
    MLP<T> n = Clone();
    n.Randomize(Scalar, rng);
    return n;
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<T> BrendanCUDA::AI::MLP::MLP<T>::Reproduce(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) const {
    MLP<T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, rng);
    return n;
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::ZeroOverwrite() {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->ZeroOverwrite();
#else
        GetLayer(i).ZeroOverwrite();
#endif
    }
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::RandomOverwrite(Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomOverwrite(rng);
#else
        GetLayer(i).RandomOverwrite(rng);
#endif
    }
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<T>::RandomOverwrite(T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomOverwrite(LowerBound, UpperBound, rng);
#else
        GetLayer(i).RandomOverwrite(LowerBound, UpperBound, rng);
#endif
    }
}
template <typename T>
__host__ void BrendanCUDA::AI::MLP::MLP<T>::Serialize(std::basic_ostream<char>& Stream) {
    Stream.write((char*)&len, sizeof(size_t) / sizeof(char));

    for (size_t i = 0; i < len; ++i) {
        GetLayer(i).Serialize(Stream);
    }
}
template <typename T>
__host__ BrendanCUDA::AI::MLP::MLP<T> BrendanCUDA::AI::MLP::MLP<T>::Deserialize(std::basic_istream<char>& Stream, activationFunction_t<T> ActivationFunction) {
    size_t n_len;
    Stream.read((char*)&n_len, sizeof(size_t) / sizeof(char));

    MLPL<T>* n_lyrs = new MLPL<T>[n_len];

    for (size_t i = 0; i < n_len; ++i) {
        n_lyrs[i] = MLPL<T>::Deserialize(Stream);
    }

    MLP<T> mlp(n_len, ActivationFunction, n_lyrs, true);

    delete[] n_lyrs;

    return mlp;
}

template BrendanCUDA::AI::MLP::MLP<float>;
template BrendanCUDA::AI::MLP::MLP<double>;