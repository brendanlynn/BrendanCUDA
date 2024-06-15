#include "brendancuda_ai_genetics_genemlp.cuh"
#include "brendancuda_random_anyrng.cuh"
#include "cuda_runtime.h"
#include "brendancuda_cudaerrorhelpers.h"

template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T>::GeneMLP(std::pair<_T*, size_t> Base, MLP::MLP<_T> Intermediate) {
    base = Base;
    intermediate = Intermediate;
}
template <typename _T>
std::pair<_T*, size_t> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Base() {
    return base;
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Dispose() {
    ThrowIfBad(cudaFree(base.first));
    intermediate.Dispose();
}
template <typename _T>
std::pair<_T*, size_t> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Run() {
    return intermediate.Run(base.first);
}
template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Clone() {
    _T* b1;
    ThrowIfBad(cudaMalloc(&b1, base.second * sizeof(_T)));
    std::pair<_T*, size_t> nb(b1, base.second);
    ThrowIfBad(cudaMemcpy(b1, base.first, base.second * sizeof(_T), cudaMemcpyDeviceToDevice));
    MLP::MLP<_T> ni = intermediate.Clone();
    return GeneMLP<_T>(nb, ni);
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar, rng);
    intermediate.Randomize(Scalar, rng);
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar, LowerBound, UpperBound, rng);
    intermediate.Randomize(Scalar, LowerBound, UpperBound, rng);
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar_Base, _T Scalar_Intermediate, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar_Base, rng);
    intermediate.Randomize(Scalar_Intermediate, rng);
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar_Base, LowerBound, UpperBound, rng);
    intermediate.Randomize(Scalar_Intermediate, LowerBound, UpperBound, rng);
}
template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar, rng);
    return n;
}
template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, rng);
    return n;
}
template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar_Base, _T Scalar_Intermediate, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, rng);
    return n;
}
template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, LowerBound, UpperBound, rng);
    return n;
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::ZeroOverwrite() {
    InitZeroArray(base.first, intermediate.InputLength());
    intermediate.ZeroOverwrite();
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::RandomOverwrite(Random::AnyRNG<uint64_t> rng) {
    InitRandomArray(base.first, intermediate.InputLength(), rng);
    intermediate.RandomOverwrite(rng);
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng) {
    InitRandomArray(base.first, intermediate.InputLength(), LowerBound, UpperBound, rng);
    intermediate.RandomOverwrite(LowerBound, UpperBound, rng);
}

template BrendanCUDA::AI::Genetics::GeneMLP<float>;
template BrendanCUDA::AI::Genetics::GeneMLP<double>;