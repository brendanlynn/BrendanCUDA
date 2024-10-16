#pragma once

#include "ai_mlp_fixedmlp.h"
#include "arrays.h"
#include "errorhelp.h"
#include "rand_randomizer.h"
#include <array>
#include <cuda_runtime.h>

namespace bcuda {
    namespace ai {
        namespace genes {
            template <typename _T, activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
            struct GeneFixedMLP final {
            private:
                using this_t = GeneFixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>;
                using mlp_t = mlp::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>;
                using inputArray_t = std::array<_T, mlp_t::InputCount()>;
                using outputArray_t = std::array<_T, mlp_t::OutputCount()>;
            public:
                inputArray_t base;
                mlp_t mlp;

                void ZeroOverwrite() {
                    rand::ClearArray<false, _T>(Span<_T>(base, mlp.InputLength()));
                    mlp.ZeroOverwrite();
                }
                template <std::uniform_random_bit_generator _TRNG>
                void RandomOverwrite(_TRNG& RNG) {
                    rand::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), RNG);
                    mlp.RandomOverwrite(RNG);
                }
                template <std::uniform_random_bit_generator _TRNG>
                void RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG) {
                    rand::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), LowerBound, UpperBound, RNG);
                    mlp.RandomOverwrite(LowerBound, UpperBound, RNG);
                }
                template <KernelCurandState _TRNG>
                __device__ void RandomOverwrite(_TRNG& RNG) {
                    rand::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), RNG);
                    mlp.RandomOverwrite(RNG);
                }
                template <KernelCurandState _TRNG>
                __device__ void RandomOverwrite(_T LowerBound, _T UpperBound, _TRNG& RNG) {
                    rand::InitRandomArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), LowerBound, UpperBound, RNG);
                    mlp.RandomOverwrite(LowerBound, UpperBound, RNG);
                }

                outputArray_t* Run() {
                    auto* outputArr = new outputArray_t;
                    Run(outputArr);
                    return outputArr;
                }
                void Run(outputArray_t* OutputArray) {
                    mlp.Run(base.data(), OutputArray->data());
                }

                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar, RNG);
                    mlp.Randomize(Scalar, RNG);
                }
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar, LowerBound, UpperBound, RNG);
                    mlp.Randomize(Scalar, LowerBound, UpperBound, RNG);
                }
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar_Base, _T Scalar_MLP, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar_Base, RNG);
                    mlp.Randomize(Scalar_MLP, RNG);
                }
                template <std::uniform_random_bit_generator _TRNG>
                void Randomize(_T Scalar_Base, _T Scalar_MLP, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar_Base, LowerBound, UpperBound, RNG);
                    mlp.Randomize(Scalar_MLP, LowerBound, UpperBound, RNG);
                }
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar, RNG);
                    return n;
                }
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar, LowerBound, UpperBound, RNG);
                    return n;
                }
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar_Base, _T Scalar_MLP, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar_Base, Scalar_MLP, RNG);
                    return n;
                }
                template <std::uniform_random_bit_generator _TRNG>
                this_t Reproduce(_T Scalar_Base, _T Scalar_MLP, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar_Base, Scalar_MLP, LowerBound, UpperBound, RNG);
                    return n;
                }
                template <KernelCurandState _TRNG>
                __device__ void Randomize(_T Scalar, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar, RNG);
                    mlp.Randomize(Scalar, RNG);
                }
                template <KernelCurandState _TRNG>
                __device__ void Randomize(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar, LowerBound, UpperBound, RNG);
                    mlp.Randomize(Scalar, LowerBound, UpperBound, RNG);
                }
                template <KernelCurandState _TRNG>
                __device__ void Randomize(_T Scalar_Base, _T Scalar_MLP, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar_Base, RNG);
                    mlp.Randomize(Scalar_MLP, RNG);
                }
                template <KernelCurandState _TRNG>
                __device__ void Randomize(_T Scalar_Base, _T Scalar_MLP, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    rand::RandomizeArray<false, _T, _TRNG>(Span<_T>(base, mlp.InputLength()), Scalar_Base, LowerBound, UpperBound, RNG);
                    mlp.Randomize(Scalar_MLP, LowerBound, UpperBound, RNG);
                }
                template <KernelCurandState _TRNG>
                __device__ this_t Reproduce(_T Scalar, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar, RNG);
                    return n;
                }
                template <KernelCurandState _TRNG>
                __device__ this_t Reproduce(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar, LowerBound, UpperBound, RNG);
                    return n;
                }
                template <KernelCurandState _TRNG>
                __device__ this_t Reproduce(_T Scalar_Base, _T Scalar_MLP, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar_Base, Scalar_MLP, RNG);
                    return n;
                }
                template <KernelCurandState _TRNG>
                __device__ this_t Reproduce(_T Scalar_Base, _T Scalar_MLP, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    this_t n = Clone();
                    n.Randomize(Scalar_Base, Scalar_MLP, LowerBound, UpperBound, RNG);
                    return n;
                }
            };
        }
    }
}