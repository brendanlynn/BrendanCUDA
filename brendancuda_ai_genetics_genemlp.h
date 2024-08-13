#pragma once

#include "brendancuda_ai_mlp_mlp.h"
#include "brendancuda_arrays.h"
#include "brendancuda_errorhelp.h"
#include "brendancuda_rand_randomizer.h"
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace AI {
        namespace Genetics {
            template <typename _T>
            class GeneMLP final {
            public:
                GeneMLP() = default;
                GeneMLP(_T* Base, MLP::MLP<_T> Intermediate);

                _T* Base();
                MLP::MLP<_T> intermediate;

                void Dispose();

                void ZeroOverwrite();
                template <std::uniform_random_bit_generator _TRNG>
                void RandomOverwrite(_TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG);

                ArrayV<_T> Run();

                GeneMLP<_T> Clone();
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                GeneMLP<_T> Reproduce(_T Scalar, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                GeneMLP<_T> Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                GeneMLP<_T> Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                GeneMLP<_T> Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG);
            private:
                _T* base;
            };
        }
    }
}

template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T>::GeneMLP(_T* Base, MLP::MLP<_T> Intermediate) {
    base = Base;
    intermediate = Intermediate;
}
template <typename _T>
_T* BrendanCUDA::AI::Genetics::GeneMLP<_T>::Base() {
    return base;
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Dispose() {
    ThrowIfBad(cudaFree(base));
    intermediate.Dispose();
}
template <typename _T>
BrendanCUDA::ArrayV<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Run() {
    return ArrayV<_T>(intermediate.Run(base), intermediate.OutputLength());
}
template <typename _T>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Clone() {
    _T* b1;
    ThrowIfBad(cudaMalloc(&b1, intermediate.InputLength() * sizeof(_T)));
    ThrowIfBad(cudaMemcpy(b1, base, intermediate.InputLength() * sizeof(_T), cudaMemcpyDeviceToDevice));
    return GeneMLP<_T>(b1, intermediate.Clone());
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar, RNG);
    intermediate.Randomize(Scalar, RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar, LowerBound, UpperBound, RNG);
    intermediate.Randomize(Scalar, LowerBound, UpperBound, RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar_Base, RNG);
    intermediate.Randomize(Scalar_Intermediate, RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::Randomize(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar_Base, LowerBound, UpperBound, RNG);
    intermediate.Randomize(Scalar_Intermediate, LowerBound, UpperBound, RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar, _TRNG& RNG) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar, RNG);
    return n;
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, RNG);
    return n;
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, RNG);
    return n;
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneMLP<_T> BrendanCUDA::AI::Genetics::GeneMLP<_T>::Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    GeneMLP<_T> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, LowerBound, UpperBound, RNG);
    return n;
}
template <typename _T>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::ZeroOverwrite() {
    Random::ClearArray<false, _T>(Span<_T>(base, intermediate.InputLength()));
    intermediate.ZeroOverwrite();
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::RandomOverwrite(_TRNG& RNG) {
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), RNG);
    intermediate.RandomOverwrite(RNG);
}
template <typename _T>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneMLP<_T>::RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), LowerBound, UpperBound, RNG);
    intermediate.RandomOverwrite(LowerBound, UpperBound, RNG);
}