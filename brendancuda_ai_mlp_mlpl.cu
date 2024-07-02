#include "brendancuda_ai_mlp_mlpl.h"
#include "brendancuda_cudaerrorhelpers.h"
#include "brendancuda_random_devicerandom.cuh"

template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<_T>::MLPL(size_t InputLength, size_t OutputLength) {
    iptLen = InputLength;
    optLen = OutputLength;
#if __CUDA_ARCH__
    wgts = (_T*)operator new[](InputLength* OutputLength * sizeof(_T));
    bias = (_T*)operator new[](OutputLength * sizeof(_T));
#else
    ThrowIfBad(cudaMalloc(&wgts, InputLength * OutputLength * sizeof(_T)));
    ThrowIfBad(cudaMalloc(&bias, OutputLength * sizeof(_T)));
#endif
}
template <typename _T>
__host__ BrendanCUDA::AI::MLP::MLPL<_T>::MLPL(size_t InputLength, size_t OutputLength, _T* Weights, _T* Bias, bool CopyFromHost)
    : MLPL(InputLength, OutputLength) {
    cudaMemcpyKind cmk = CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    ThrowIfBad(cudaMemcpy(wgts, Weights, InputLength * OutputLength * sizeof(_T), cmk));
    ThrowIfBad(cudaMemcpy(bias, Bias, OutputLength * sizeof(_T), cmk));
}
template <typename _T>
__device__ BrendanCUDA::AI::MLP::MLPL<_T>::MLPL(size_t InputLength, size_t OutputLength, _T* Weights, _T* Bias)
    : MLPL(InputLength, OutputLength) {
    deviceMemcpy(wgts, Weights, InputLength * OutputLength * sizeof(_T));
    deviceMemcpy(bias, Bias, OutputLength * sizeof(_T));
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::ZeroOverwrite() {
    InitZeroArray(Span<_T>(wgts, iptLen * optLen));
    InitZeroArray(Span<_T>(bias, optLen));
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::RandomOverwrite(Random::AnyRNG<uint64_t> RNG) {
    InitRandomArray(Span<_T>(wgts, iptLen * optLen), RNG);
    InitRandomArray(Span<_T>(bias, optLen), RNG);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) {
    InitRandomArray(Span<_T>(wgts, iptLen * optLen), LowerBound, UpperBound, RNG);
    InitRandomArray(Span<_T>(bias, optLen), LowerBound, UpperBound, RNG);
}
template <typename _T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLPL<_T>::InputLength() const {
    return iptLen;
}
template <typename _T>
__host__ __device__ size_t BrendanCUDA::AI::MLP::MLPL<_T>::OutputLength() const {
    return optLen;
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::Weights() const {
    return wgts;
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::Bias() const {
    return bias;
}
template <typename _T>
__host__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::Run(_T* Input) const {
    _T* output;
    ThrowIfBad(cudaMalloc(&output, sizeof(_T) * optLen));
    ThrowIfBad(cudaMemcpy(output, bias, sizeof(float) * optLen, cudaMemcpyDeviceToDevice));

    cublasHandle_t h;
    cublasCreate(&h);

    if constexpr (std::is_same<float, _T>::value) {
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
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::Dispose() {
#if __CUDA_ARCH__
    delete[] wgts;
    delete[] bias;
#else
    ThrowIfBad(cudaFree(wgts));
    ThrowIfBad(cudaFree(bias));
#endif
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::Weight(size_t Index) const {
    return &wgts[Index];
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::Weight(size_t X, size_t Y) const {
    return &wgts[X * optLen + Y];
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::Bias(size_t Index) const {
    return &bias[Index];
}
template <typename _T>
__host__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetWeights(bool CopyToHost) const {
    size_t l = iptLen * optLen * sizeof(_T);
    if (CopyToHost) {
        _T* output = (_T*)operator new[](l);
        ThrowIfBad(cudaMemcpy(output, wgts, l, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        _T* output;
        ThrowIfBad(cudaMalloc(&output, l));
        ThrowIfBad(cudaMemcpy(output, wgts, l, cudaMemcpyDeviceToHost));
        return output;
    }
}
template <typename _T>
__device__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetWeights() const {
    size_t l = iptLen * optLen * sizeof(_T);
    _T* output = (_T*)operator new[](l);
    deviceMemcpy(output, wgts, l);
    return output;
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::AI::MLP::MLPL<_T>::GetWeight(size_t Index) {
#if __CUDA_ARCH__
    return wgts[Index];
#else
    _T v;
    ThrowIfBad(cudaMemcpy(&v, Weight(Index), sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::AI::MLP::MLPL<_T>::GetWeight(size_t X, size_t Y) {
#if __CUDA_ARCH__
    return *Weight(X, Y);
#else
    _T v;
    ThrowIfBad(cudaMemcpy(&v, Weight(X, Y), sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename _T>
__host__ void BrendanCUDA::AI::MLP::MLPL<_T>::SetWeights(_T* Values, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(wgts, Values, iptLen * optLen * sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::AI::MLP::MLPL<_T>::SetWeights(_T* Values) {
    deviceMemcpy(wgts, Values, iptLen * optLen * sizeof(_T));
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::SetWeight(_T Value, size_t Index) {
#if __CUDA_ARCH__
    wgts[Index] = Value;
#else
    ThrowIfBad(cudaMemcpy(Weight(Index), &Value, sizeof(_T), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::SetWeight(_T Value, size_t X, size_t Y) {
#if __CUDA_ARCH__
    *Weight(X, Y) = Value;
#else
    ThrowIfBad(cudaMemcpy(Weight(X, Y), &Value, sizeof(_T), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
__host__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetBias(bool CopyToHost) const {
    size_t l = optLen * sizeof(_T);
    if (CopyToHost) {
        _T* output = (_T*)operator new[](l);
        ThrowIfBad(cudaMemcpy(output, bias, l, cudaMemcpyDeviceToHost));
        return output;
    }
    else {
        _T* output;
        ThrowIfBad(cudaMalloc(&output, l));
        ThrowIfBad(cudaMemcpy(output, bias, l, cudaMemcpyDeviceToHost));
        return output;
    }
}
template <typename _T>
__device__ _T* BrendanCUDA::AI::MLP::MLPL<_T>::GetBias() const {
    size_t l = optLen * sizeof(_T);
    _T* output = (_T*)operator new[](l);
    deviceMemcpy(output, bias, l);
    return output;
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::AI::MLP::MLPL<_T>::GetBias(size_t Index) const {
#if __CUDA_ARCH__
    return bias[Index];
#else
    _T v;
    ThrowIfBad(cudaMemcpy(&v, Bias(Index), sizeof(MLPL<_T>), cudaMemcpyDeviceToHost));
    return v;
#endif
}
template <typename _T>
__host__ void BrendanCUDA::AI::MLP::MLPL<_T>::SetBias(_T* Values, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(bias, Values, optLen * sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::AI::MLP::MLPL<_T>::SetBias(_T* Values) {
    deviceMemcpy(bias, Values, optLen * sizeof(_T));
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::SetBias(_T Value, size_t Index) {
#if __CUDA_ARCH__
    bias[Index] = Value;
#else
    ThrowIfBad(cudaMemcpy(Bias(Index), &Value, sizeof(_T), cudaMemcpyHostToDevice));
#endif
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLPL<_T>::Clone() const {
#if __CUDA_ARCH__
    return MLPL<_T>(iptLen, optLen, wgts, bias);
#else
    return MLPL<_T>(iptLen, optLen, wgts, bias, false);
#endif
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::Randomize(_T Scalar, Random::AnyRNG<uint64_t> RNG) {
    RandomizeArray(Span<_T>(wgts, iptLen * optLen), Scalar, RNG);
    RandomizeArray(Span<_T>(bias, optLen), Scalar, RNG);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::AI::MLP::MLPL<_T>::Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) {
    RandomizeArray(Span<_T>(wgts, iptLen * optLen), Scalar, LowerBound, UpperBound, RNG);
    RandomizeArray(Span<_T>(bias, optLen), Scalar, LowerBound, UpperBound, RNG);
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLPL<_T>::Reproduce(_T Scalar, Random::AnyRNG<uint64_t> RNG) const {
    MLPL<_T> n = Clone();
    n.Randomize(Scalar, RNG);
    return n;
}
template <typename _T>
__host__ __device__ BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLPL<_T>::Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) const {
    MLPL<_T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, RNG);
    return n;
}
template <typename _T>
__host__ void BrendanCUDA::AI::MLP::MLPL<_T>::Serialize(std::basic_ostream<char>& Stream) {
    constexpr size_t s0 = sizeof(size_t) / sizeof(char);
    static_assert(!(sizeof(size_t) % sizeof(char)), "The size (in bytes) of type std::size_t must be evenly divisible by the size (in bytes) of type char.");
    constexpr size_t s1 = sizeof(_T) / sizeof(char);
    static_assert(!(sizeof(_T) % sizeof(char)), "The size (in bytes) of type _T must be evenly divisible by the size (in bytes) of type char.");

    _T* weights = GetWeights(true);
    _T* bias = GetBias(true);

    Stream.write((char*)&iptLen, s0);
    Stream.write((char*)&optLen, s0);

    Stream.write((char*)weights, iptLen * optLen * s1);
    Stream.write((char*)bias, optLen * s1);

    delete[] weights;
    delete[] bias;
}
template <typename _T>
__host__ BrendanCUDA::AI::MLP::MLPL<_T> BrendanCUDA::AI::MLP::MLPL<_T>::Deserialize(std::basic_istream<char>& Stream) {
    constexpr size_t s0 = sizeof(size_t) / sizeof(char);
    static_assert(!(sizeof(size_t) % sizeof(char)), "The size (in bytes) of type std::size_t must be evenly divisible by the size (in bytes) of type char.");
    constexpr size_t s1 = sizeof(_T) / sizeof(char);
    static_assert(!(sizeof(_T) % sizeof(char)), "The size (in bytes) of type _T must be evenly divisible by the size (in bytes) of type char.");

    size_t n_iptLen;
    size_t n_optLen;

    Stream.read((char*)&n_iptLen, s0);
    Stream.read((char*)&n_optLen, s0);

    _T* n_weights = new _T[n_iptLen * n_optLen];
    _T* n_bias = new _T[n_optLen];

    Stream.read((char*)n_weights, n_iptLen * n_optLen * s1);
    Stream.read((char*)n_bias, n_optLen * s1);

    MLPL<_T> mlpl(n_iptLen, n_optLen, n_weights, n_bias, true);

    delete[] n_weights;
    delete[] n_bias;

    return mlpl;
}

template BrendanCUDA::AI::MLP::MLPL<float>;
template BrendanCUDA::AI::MLP::MLPL<double>;