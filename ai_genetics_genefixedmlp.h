#pragma once

#include "ai_mlp_fixedmlp.h"
#include "arrays.h"
#include "errorhelp.h"
#include "rand_randomizer.h"
#include <array>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace AI {
        namespace Genetics {
            template <typename _T, activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
            struct GeneFixedMLP final {
            private:
                using this_t = GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>;
                using mlp_t = MLP::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>;
                using inputArray_t = std::array<_T, mlp_t::InputCount()>;
                using outputArray_t = std::array<_T, mlp_t::OutputCount()>;
            public:
                inputArray_t base;
                mlp_t mlp;

                void ZeroOverwrite();
                template <std::uniform_random_bit_generator _TRNG>
                void RandomOverwrite(_TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG);

                outputArray_t* Run();
                void Run(outputArray_t* OutputArray);

                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG);
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG);
            };
        }
    }
}

template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
auto BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Run() -> outputArray_t* {
    auto* outputArr = new outputArray_t;
    Run(outputArr);
    return outputArr;
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Run(outputArray_t* OutputArray) {
    mlp.Run(base.data(), OutputArray->data());
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Randomize(_T Scalar, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar, RNG);
    intermediate.Randomize(Scalar, RNG);
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar, LowerBound, UpperBound, RNG);
    intermediate.Randomize(Scalar, LowerBound, UpperBound, RNG);
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Randomize(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar_Base, RNG);
    intermediate.Randomize(Scalar_Intermediate, RNG);
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Randomize(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), Scalar_Base, LowerBound, UpperBound, RNG);
    intermediate.Randomize(Scalar_Intermediate, LowerBound, UpperBound, RNG);
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Reproduce(_T Scalar, _TRNG& RNG) {
    GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> n = Clone();
    n.Randomize(Scalar, RNG);
    return n;
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> n = Clone();
    n.Randomize(Scalar, LowerBound, UpperBound, RNG);
    return n;
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _TRNG& RNG) {
    GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, RNG);
    return n;
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, _TRNG& RNG) {
    GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...> n = Clone();
    n.Randomize(Scalar_Base, Scalar_Intermediate, LowerBound, UpperBound, RNG);
    return n;
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::ZeroOverwrite() {
    Random::ClearArray<false, _T>(Span<_T>(base, intermediate.InputLength()));
    intermediate.ZeroOverwrite();
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::RandomOverwrite(_TRNG& RNG) {
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), RNG);
    intermediate.RandomOverwrite(RNG);
}
template <typename _T, BrendanCUDA::AI::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
template <std::uniform_random_bit_generator _TRNG>
void BrendanCUDA::AI::Genetics::GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>::RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG) {
    Random::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, intermediate.InputLength()), LowerBound, UpperBound, RNG);
    intermediate.RandomOverwrite(LowerBound, UpperBound, RNG);
}