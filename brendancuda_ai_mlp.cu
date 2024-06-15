#include "brendancuda_ai_mlp.cuh"
#include "brendancuda_cudaerrorhelpers.h"

template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<_T>::MLP(size_t Length, activationFunction_t<_T> ActivationFunction) {
    len = Length;
    actnFunc = ActivationFunction;
#if __CUDA_ARCH__
    lyrs = new MLPL<_T>[Length];
#else
    ThrowIfBad(cudaMalloc(&lyrs, Length * sizeof(MLPL<_T>)));
#endif
}
template <typename _T>
__host__ BrendanCUDA::AI::MLP::MLP<_T>::MLP(size_t Length, activationFunction_t<_T> ActivationFunction, MLPL<_T>* Layers, bool CopyFromHost)
    : MLP(Length, ActivationFunction) {
    ThrowIfBad(cudaMemcpy(lyrs, Layers, Length * sizeof(MLPL<_T>), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ BrendanCUDA::AI::MLP::MLP<_T>::MLP(size_t Length, activationFunction_t<_T> ActivationFunction, MLPL<_T>* Layers)
    : MLP(Length, ActivationFunction) {
    deviceMemcpy(lyrs, Layers, Length * sizeof(MLPL<_T>));
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::Dispose() {
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
    ThrowIfBad(cudaFree(lyrs));
#endif
}
template <typename _T>
__host__ std::pair<_T*, size_t> BrendanCUDA::AI::MLP::MLP<_T>::Run(_T* Input) {
    if (len == 0) {
        return std::pair<_T*, size_t>(0, 0);
    }
#if __CUDA_ARCH__
    MLPL<_T>* l = Layer(0);
    Input = l->Run(Input);
    RunActivationFunctionOnArray(Input, l->OutputLength());
#else
    MLPL<_T> l = GetLayer(0);
    Input = l.Run(Input);
    RunActivationFunctionOnArray(Input, l.OutputLength());
#endif
    for (size_t i = 1; i < len; ++i) {
#if __CUDA_ARCH__
        l = Layer(i);
        _T* nxt = l->Run(Input);
        RunActivationFunctionOnArray(nxt, l->OutputLength());
        delete[] Input;
#else
        l = GetLayer(i);
        _T* nxt = l.Run(Input);
        RunActivationFunctionOnArray(nxt, l.OutputLength());
        ThrowIfBad(cudaFree(Input));
#endif
        Input = nxt;
    }
    return std::pair<_T*, size_t>(Input, OutputLength());
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::Layers() const {
    return lyrs;
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::Layer(size_t LayerIndex) const {
    return &lyrs[LayerIndex];
}
template <typename _T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLP<_T>::LayerCount() const {
    return len;
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::activationFunction_t<_T> BrendanCUDA::AI::MLP::MLP<_T>::ActivationFunction() const {
    return actnFunc;
}
template <typename _T>
__host__ BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::GetLayers(bool CopyToHost) const {
    MLPL<_T>* p;
    if (CopyToHost) {
        p = new MLPL<_T>[len];
        ThrowIfBad(cudaMemcpy(p, lyrs, sizeof(MLPL<_T>) * len, cudaMemcpyDeviceToHost));
    }
    else {
        ThrowIfBad(cudaMalloc(&p, sizeof(MLPL<_T>) * len));
        ThrowIfBad(cudaMemcpy(p, lyrs, sizeof(MLPL<_T>) * len, cudaMemcpyDeviceToDevice));
    }
    return p;
}
template <typename _T>
__device__ BrendanCUDA::AI::MLP::MLPL<_T>* BrendanCUDA::AI::MLP::MLP<_T>::GetLayers() const {
    MLPL<_T>* p = new MLPL<_T>[len];
    deviceMemcpy(p, lyrs, sizeof(MLPL<_T>) * len);
    return p;
}
template <typename _T>
__host__ void BrendanCUDA::AI::MLP::MLP<_T>::SetLayers(MLPL<_T>* Layers, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(lyrs, Layers, sizeof(MLPL<_T>) * len, CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::AI::MLP::MLP<_T>::SetLayers(MLPL<_T>* Layers) {
    deviceMemcpy(lyrs, Layers, sizeof(MLPL<_T>) * len);
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLP<_T>::GetLayer(size_t LayerIndex) const {
#if __CUDA_ARCH__
    return lyrs[LayerIndex];
#else
    MLPL<_T> r;
    ThrowIfBad(cudaMemcpy(&r, &lyrs[LayerIndex], sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return r;
#endif
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::SetLayer(size_t LayerIndex, MLPL<_T> Layer) {
#if __CUDA_ARCH__
    lyrs[LayerIndex] = Layer;
#else
    ThrowIfBad(cudaMemcpy(&lyrs[LayerIndex], &Layer, sizeof(MLPL<_T>), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLP<_T>::InputLength() {
    if (len == 0) {
        return 0;
    }
#if __CUDA_ARCH__
    return Layer(0)->InputLength();
#else
    return GetLayer(0).InputLength();
#endif
}
template <typename _T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLP<_T>::OutputLength() {
    if (len == 0) {
        return 0;
    }
#if __CUDA_ARCH__
    return Layer(len - 1)->OutputLength();
#else
    return GetLayer(len - 1).OutputLength();
#endif
}
__global__ void runActivationFunctionOnArrayKernel(float* Array, BrendanCUDA::AI::activationFunction_t<float> ActivationFunction) {
    float& p(Array[blockIdx.x]);
    p = ActivationFunction(p);
}
__global__ void runActivationFunctionOnArrayKernel(double* Array, BrendanCUDA::AI::activationFunction_t<double> ActivationFunction) {
    double& p(Array[blockIdx.x]);
    p = ActivationFunction(p);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::RunActivationFunctionOnArray(_T* Array, size_t Length) {
    activationFunction_t<_T> af = actnFunc;
#if __CUDA_ARCH__
    for (size_t i = 0; i < Length; ++i) {
        _T& p(Array[i]);
        p = af(p);
    }
#else
    runActivationFunctionOnArrayKernel<<<Length, 1>>>(Array, af);
#endif
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<_T> BrendanCUDA::AI::MLP::MLP<_T>::Clone() const {
    MLP<_T> m(len, actnFunc);
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        m.SetLayer(i, Layer(i)->Clone());
#else
        m.SetLayer(i, GetLayer(i).Clone());
#endif
    }
    return m;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::Randomize(_T Scalar, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->Randomize(Scalar, rng);
#else
        GetLayer(i).Randomize(Scalar, rng);
#endif
    }
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->Randomize(Scalar, LowerBound, UpperBound, rng);
#else
        GetLayer(i).Randomize(Scalar, LowerBound, UpperBound, rng);
#endif
    }
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<_T> BrendanCUDA::AI::MLP::MLP<_T>::Reproduce(_T Scalar, Random::AnyRNG<uint64_t> rng) const {
    MLP<_T> n = Clone();
    n.Randomize(Scalar, rng);
    return n;
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLP<_T> BrendanCUDA::AI::MLP::MLP<_T>::Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) const {
    MLP<_T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, rng);
    return n;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::ZeroOverwrite() {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->ZeroOverwrite();
#else
        GetLayer(i).ZeroOverwrite();
#endif
    }
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::RandomOverwrite(Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomOverwrite(rng);
#else
        GetLayer(i).RandomOverwrite(rng);
#endif
    }
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLP<_T>::RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) {
    for (size_t i = 0; i < len; ++i) {
#if __CUDA_ARCH__
        Layer(i)->RandomOverwrite(LowerBound, UpperBound, rng);
#else
        GetLayer(i).RandomOverwrite(LowerBound, UpperBound, rng);
#endif
    }
}
template <typename _T>
__host__ void BrendanCUDA::AI::MLP::MLP<_T>::Serialize(std::basic_ostream<char>& Stream) {
    Stream.write((char*)&len, sizeof(size_t) / sizeof(char));

    for (size_t i = 0; i < len; ++i) {
        GetLayer(i).Serialize(Stream);
    }
}
template <typename _T>
__host__ BrendanCUDA::AI::MLP::MLP<_T> BrendanCUDA::AI::MLP::MLP<_T>::Deserialize(std::basic_istream<char>& Stream, activationFunction_t<_T> ActivationFunction) {
    size_t n_len;
    Stream.read((char*)&n_len, sizeof(size_t) / sizeof(char));

    MLPL<_T>* n_lyrs = new MLPL<_T>[n_len];

    for (size_t i = 0; i < n_len; ++i) {
        n_lyrs[i] = MLPL<_T>::Deserialize(Stream);
    }

    MLP<_T> mlp(n_len, ActivationFunction, n_lyrs, true);

    delete[] n_lyrs;

    return mlp;
}

template BrendanCUDA::AI::MLP::MLP<float>;
template BrendanCUDA::AI::MLP::MLP<double>;