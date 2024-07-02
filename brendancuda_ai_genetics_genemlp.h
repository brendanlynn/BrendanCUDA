#pragma once

#include "brendancuda_ai_mlp.h"
#include "brendancuda_arrays.h"
#include <vector>

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
                void RandomOverwrite(Random::AnyRNG<uint64_t> RNG);
                void RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);

                ArrayV<_T> Run();

                GeneMLP<_T> Clone();
                void Randomize(_T Scalar, Random::AnyRNG<uint64_t> RNG);
                void Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, Random::AnyRNG<uint64_t> RNG);
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);
                GeneMLP<_T> Reproduce(_T Scalar, Random::AnyRNG<uint64_t> RNG);
                GeneMLP<_T> Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);
                GeneMLP<_T> Reproduce(_T Scalar_Base, _T Scalar_Intermediate, Random::AnyRNG<uint64_t> RNG);
                GeneMLP<_T> Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);
            private:
                _T* base;
            };
        }
    }
}