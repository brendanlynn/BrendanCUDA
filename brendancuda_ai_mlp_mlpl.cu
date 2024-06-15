#include "brendancuda_ai_mlp_mlpl.cuh"
#include "brendancuda_cudaerrorhelpers.h"

template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<T>::MLPL(size_t InputLength, size_t OutputLength) {
    iptLen = InputLength;
    optLen = OutputLength;
#if __CUDA_ARCH__
    wgts = (T*)operator new[](InputLength* OutputLength * sizeof(T));
    bias = (T*)operator new[](OutputLength * sizeof(T));
#else
    ThrowIfBad(cudaMalloc(&wgts, InputLength * OutputLength * sizeof(T)));
    ThrowIfBad(cudaMalloc(&bias, OutputLength * sizeof(T)));
#endif
}
template <typename T>
__host__ BrendanCUDA::AI::MLP::MLPL<T>::MLPL(size_t InputLength, size_t OutputLength, T* Weights, T* Bias, bool CopyFromHost)
    : MLPL(InputLength, OutputLength) {
    cudaMemcpyKind cmk = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(wgts, Weights, InputLength * OutputLength * sizeof(T), cmk));
    ThrowIfBad(cudaMemcpy(bias, Bias, OutputLength * sizeof(T), cmk));
}
template <typename T>
__device__ BrendanCUDA::AI::MLP::MLPL<T>::MLPL(size_t InputLength, size_t OutputLength, T* Weights, T* Bias)
    : MLPL(InputLength, OutputLength) {
    deviceMemcpy(wgts, Weights, InputLength * OutputLength * sizeof(T));
    deviceMemcpy(bias, Bias, OutputLength * sizeof(T));
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::ZeroOverwrite() {
    InitZeroArray(wgts, iptLen * optLen);
    InitZeroArray(bias, optLen);
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::RandomOverwrite(Random::AnyRNG<uint64_t> rng) {
    InitRandomArray(wgts, iptLen * optLen, rng);
    InitRandomArray(bias, optLen, rng);
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::RandomOverwrite(T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    InitRandomArray(wgts, iptLen * optLen, LowerBound, UpperBound, rng);
    InitRandomArray(bias, optLen, LowerBound, UpperBound, rng);
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLPL<T>::InputLength() const {
    return iptLen;
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLPL<T>::OutputLength() const {
    return optLen;
}
template <typename T>
__host__ __device__ T* BrendanCUDA::AI::MLP::MLPL<T>::Weights() const {
    return wgts;
}
template <typename T>
__host__ __device__ T* BrendanCUDA::AI::MLP::MLPL<T>::Bias() const {
    return bias;
}
template <typename T>
__host__ T* BrendanCUDA::AI::MLP::MLPL<T>::Run(T* Input) const {
    T* output;
    ThrowIfBad(cudaMalloc(&output, sizeof(T) * optLen));
    ThrowIfBad(cudaMemcpy(output, bias, sizeof(float) * optLen, cudaMemcpyDeviceToDevice));

    cublasHandle_t h;
    cublasCreate(&h);

    if (std::is_same<float, T>::value) {
        float f = 1.0f;
        cublasSgemv(h, CUBLAS_OP_N, optLen, iptLen, &f, (float*)wgts, optLen, (float*)Input, 1, &f, (float*)output, 1);
    }
    else {
        double d = 1.0;
        cublasDgemv(h, CUBLAS_OP_N, optLen, iptLen, &d, (double*)wgts, optLen, (double*)Input, 1, &d, (double*)output, 1);
    }

    cublasDestroy(h);

    return output;
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::Dispose() {
#if __CUDA_ARCH__
    delete[] wgts;
    delete[] bias;
#else
    ThrowIfBad(cudaFree(wgts));
    ThrowIfBad(cudaFree(bias));
#endif
}
template <typename T>
__host__ __device__ T* BrendanCUDA::AI::MLP::MLPL<T>::Weight(size_t Index) const {
    return &wgts[Index];
}
template <typename T>
__host__ __device__ T* BrendanCUDA::AI::MLP::MLPL<T>::Weight(size_t X, size_t Y) const {
    return &wgts[X * optLen + Y];
}
template <typename T>
__host__ __device__ T* BrendanCUDA::AI::MLP::MLPL<T>::Bias(size_t Index) const {
    return &bias[Index];
}
template <typename T>
__host__ T* BrendanCUDA::AI::MLP::MLPL<T>::GetWeights(bool CopyToHost) const {
    size_t l = iptLen * optLen * sizeof(T);
    if (CopyToHost) {
        T* output = (T*)operator new[](l);
        ThrowIfBad(cudaMemcpy(output, wgts, l, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        T* output;
        ThrowIfBad(cudaMalloc(&output, l));
        ThrowIfBad(cudaMemcpy(output, wgts, l, cudaMemcpyDeviceToHost));
        return output;
    }
}
template <typename T>
__device__ T* BrendanCUDA::AI::MLP::MLPL<T>::GetWeights() const {
    size_t l = iptLen * optLen * sizeof(T);
    T* output = (T*)operator new[](l);
    deviceMemcpy(output, wgts, l);
    return output;
}
template <typename T>
__host__ __device__ T BrendanCUDA::AI::MLP::MLPL<T>::GetWeight(size_t Index) {
#if __CUDA_ARCH__
    return wgts[Index];
#else
    T v;
    ThrowIfBad(cudaMemcpy(&v, Weight(Index), sizeof(MLPL<T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename T>
__host__ __device__ T BrendanCUDA::AI::MLP::MLPL<T>::GetWeight(size_t X, size_t Y) {
#if __CUDA_ARCH__
    return *Weight(X, Y);
#else
    T v;
    ThrowIfBad(cudaMemcpy(&v, Weight(X, Y), sizeof(MLPL<T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename T>
__host__ void BrendanCUDA::AI::MLP::MLPL<T>::SetWeights(T* Values, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(wgts, Values, iptLen * optLen * sizeof(T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename T>
__device__ void BrendanCUDA::AI::MLP::MLPL<T>::SetWeights(T* Values) {
    deviceMemcpy(wgts, Values, iptLen * optLen * sizeof(T));
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::SetWeight(T Value, size_t Index) {
#if __CUDA_ARCH__
    wgts[Index] = Value;
#else
    ThrowIfBad(cudaMemcpy(Weight(Index), &Value, sizeof(T), cudaMemcpyHostToDevice));
#endif
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::SetWeight(T Value, size_t X, size_t Y) {
#if __CUDA_ARCH__
    *Weight(X, Y) = Value;
#else
    ThrowIfBad(cudaMemcpy(Weight(X, Y), &Value, sizeof(T), cudaMemcpyHostToDevice));
#endif
}
template <typename T>
__host__ T* BrendanCUDA::AI::MLP::MLPL<T>::GetBias(bool CopyToHost) const {
    size_t l = optLen * sizeof(T);
    if (CopyToHost) {
        T* output = (T*)operator new[](l);
        ThrowIfBad(cudaMemcpy(output, bias, l, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        T* output;
        ThrowIfBad(cudaMalloc(&output, l));
        ThrowIfBad(cudaMemcpy(output, bias, l, cudaMemcpyDeviceToHost));
        return output;
    }
}
template <typename T>
__device__ T* BrendanCUDA::AI::MLP::MLPL<T>::GetBias() const {
    size_t l = optLen * sizeof(T);
    T* output = (T*)operator new[](l);
    deviceMemcpy(output, bias, l);
    return output;
}
template <typename T>
__host__ __device__ T BrendanCUDA::AI::MLP::MLPL<T>::GetBias(size_t Index) const {
#if __CUDA_ARCH__
    return bias[Index];
#else
    T v;
    ThrowIfBad(cudaMemcpy(&v, Bias(Index), sizeof(MLPL<T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename T>
__host__ void BrendanCUDA::AI::MLP::MLPL<T>::SetBias(T* Values, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(bias, Values, optLen * sizeof(T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename T>
__device__ void BrendanCUDA::AI::MLP::MLPL<T>::SetBias(T* Values) {
    deviceMemcpy(bias, Values, optLen * sizeof(T));
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::SetBias(T Value, size_t Index) {
#if __CUDA_ARCH__
    bias[Index] = Value;
#else
    ThrowIfBad(cudaMemcpy(Bias(Index), &Value, sizeof(T), cudaMemcpyHostToDevice));
#endif
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<T> BrendanCUDA::AI::MLP::MLPL<T>::Clone() const {
#if __CUDA_ARCH__
    return MLPL<T>(iptLen, optLen, wgts, bias);
#else
    return MLPL<T>(iptLen, optLen, wgts, bias, false);
#endif
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::Randomize(T Scalar, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(wgts, iptLen * optLen, Scalar, rng);
#if defined(_DEBUG) && !defined(__CUDA_ARCH__)
    auto x = cudaDeviceSynchronize();
#endif
    RandomizeArray(bias, optLen, Scalar, rng);
#if defined(_DEBUG) && !defined(__CUDA_ARCH__)
    auto y = cudaDeviceSynchronize();
#endif
}
template <typename T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<T>::Randomize(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(wgts, iptLen * optLen, Scalar, LowerBound, UpperBound, rng);
#if defined(_DEBUG) && !defined(__CUDA_ARCH__)
    auto x = cudaDeviceSynchronize();
#endif
    RandomizeArray(bias, optLen, Scalar, LowerBound, UpperBound, rng);
#if defined(_DEBUG) && !defined(__CUDA_ARCH__)
    auto y = cudaDeviceSynchronize();
#endif
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<T> BrendanCUDA::AI::MLP::MLPL<T>::Reproduce(T Scalar, Random::AnyRNG<uint64_t> rng) const {
    MLPL<T> n = Clone();
    n.Randomize(Scalar, rng);
    return n;
}
template <typename T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<T> BrendanCUDA::AI::MLP::MLPL<T>::Reproduce(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) const {
    MLPL<T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, rng);
    return n;
}
template <typename T>
__host__ void BrendanCUDA::AI::MLP::MLPL<T>::Serialize(std::basic_ostream<char>& Stream) {
    constexpr size_t s0 = sizeof(size_t) / sizeof(char);
    static_assert(!(sizeof(size_t) % sizeof(char)), "The size (in bytes) of type std::size_t must be evenly divisible by the size (in bytes) of type char.");
    constexpr size_t s1 = sizeof(T) / sizeof(char);
    static_assert(!(sizeof(T) % sizeof(char)), "The size (in bytes) of type T must be evenly divisible by the size (in bytes) of type char.");

    T* weights = GetWeights(true);
    T* bias = GetBias(true);

    Stream.write((char*)&iptLen, s0);
    Stream.write((char*)&optLen, s0);

    Stream.write((char*)weights, iptLen * optLen * s1);
    Stream.write((char*)bias, optLen * s1);

    delete[] weights;
    delete[] bias;
}
template <typename T>
__host__ BrendanCUDA::AI::MLP::MLPL<T> BrendanCUDA::AI::MLP::MLPL<T>::Deserialize(std::basic_istream<char>& Stream) {
    constexpr size_t s0 = sizeof(size_t) / sizeof(char);
    static_assert(!(sizeof(size_t) % sizeof(char)), "The size (in bytes) of type std::size_t must be evenly divisible by the size (in bytes) of type char.");
    constexpr size_t s1 = sizeof(T) / sizeof(char);
    static_assert(!(sizeof(T) % sizeof(char)), "The size (in bytes) of type T must be evenly divisible by the size (in bytes) of type char.");

    size_t n_iptLen;
    size_t n_optLen;

    Stream.read((char*)&n_iptLen, s0);
    Stream.read((char*)&n_optLen, s0);

    T* n_weights = new T[n_iptLen * n_optLen];
    T* n_bias = new T[n_optLen];

    Stream.read((char*)n_weights, n_iptLen * n_optLen * s1);
    Stream.read((char*)n_bias, n_optLen * s1);

    MLPL<T> mlpl(n_iptLen, n_optLen, n_weights, n_bias, true);

    delete[] n_weights;
    delete[] n_bias;

    return mlpl;
}

template BrendanCUDA::AI::MLP::MLPL<float>;
template BrendanCUDA::AI::MLP::MLPL<double>;