#include "brendancuda_ai_genetics_genemlp.cuh"
#include "brendancuda_random_anyrng.cuh"
#include "cuda_runtime.h"
#include "brendancuda_cudaerrorhelpers.h"

template <typename T>
BrendanCUDA::AI::Genetics::GeneMLP<T>::GeneMLP(std::pair<T*, size_t> Base, MLP::MLP<T> Intermediate) {
    base = Base;
    this->Intermediate = Intermediate;
}
template <typename T>
std::pair<T*, size_t> BrendanCUDA::AI::Genetics::GeneMLP<T>::Base() {
    return base;
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::Dispose() {
    ThrowIfBad(cudaFree(base.first));
    Intermediate.Dispose();
}
template <typename T>
std::pair<T*, size_t> BrendanCUDA::AI::Genetics::GeneMLP<T>::Run() {
    return Intermediate.Run(base.first);
}
template <typename T>
BrendanCUDA::AI::Genetics::GeneMLP<T> BrendanCUDA::AI::Genetics::GeneMLP<T>::Clone() {
    T* b1;
    ThrowIfBad(cudaMalloc(&b1, base.second * sizeof(T)));
    std::pair<T*, size_t> nb(b1, base.second);
    ThrowIfBad(cudaMemcpy(b1, base.first, base.second * sizeof(T), cudaMemcpyDeviceToDevice));
    MLP::MLP<T> ni = Intermediate.Clone();
    return GeneMLP<T>(nb, ni);
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::Randomize(T Scalar, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar, rng);
    Intermediate.Randomize(Scalar, rng);
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::Randomize(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar, LowerBound, UpperBound, rng);
    Intermediate.Randomize(Scalar, LowerBound, UpperBound, rng);
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::Randomize(T Scalar_Base, T Scalar_Intermediate, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar_Base, rng);
    Intermediate.Randomize(Scalar_Intermediate, rng);
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::Randomize(T Scalar_Base, T Scalar_Intermediate, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    RandomizeArray(base.first, base.second, Scalar_Base, LowerBound, UpperBound, rng);
    Intermediate.Randomize(Scalar_Intermediate, LowerBound, UpperBound, rng);
}
template <typename T>
BrendanCUDA::AI::Genetics::GeneMLP<T> BrendanCUDA::AI::Genetics::GeneMLP<T>::Reproduce(T Scalar, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<T> n = Clone();
    n.Randomize(Scalar, rng);
    return n;
}
template <typename T>
BrendanCUDA::AI::Genetics::GeneMLP<T> BrendanCUDA::AI::Genetics::GeneMLP<T>::Reproduce(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, rng);
    return n;
}
template <typename T>
BrendanCUDA::AI::Genetics::GeneMLP<T> BrendanCUDA::AI::Genetics::GeneMLP<T>::Reproduce(T Scalar_Base, T Scalar_Intermediate, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<T> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, rng);
    return n;
}
template <typename T>
BrendanCUDA::AI::Genetics::GeneMLP<T> BrendanCUDA::AI::Genetics::GeneMLP<T>::Reproduce(T Scalar_Base, T Scalar_Intermediate, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    GeneMLP<T> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, LowerBound, UpperBound, rng);
    return n;
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::ZeroOverwrite() {
    InitZeroArray(base.first, Intermediate.InputLength());
    Intermediate.ZeroOverwrite();
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::RandomOverwrite(Random::AnyRNG<uint64_t> rng) {
    InitRandomArray(base.first, Intermediate.InputLength(), rng);
    Intermediate.RandomOverwrite(rng);
}
template <typename T>
void BrendanCUDA::AI::Genetics::GeneMLP<T>::RandomOverwrite(T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) {
    InitRandomArray(base.first, Intermediate.InputLength(), LowerBound, UpperBound, rng);
    Intermediate.RandomOverwrite(LowerBound, UpperBound, rng);
}

template BrendanCUDA::AI::Genetics::GeneMLP<float>;
template BrendanCUDA::AI::Genetics::GeneMLP<double>;