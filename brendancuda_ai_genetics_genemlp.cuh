#pragma once

#include "brendancuda_ai_mlp.cuh"
#include <vector>

namespace BrendanCUDA {
    namespace AI {
        namespace Genetics {
            template <typename _T>
            class GeneMLP final {
            public:
                GeneMLP() = default;
                GeneMLP(std::pair<_T*, size_t> Base, MLP::MLP<_T> Intermediate);

                std::pair<_T*, size_t> Base();
                MLP::MLP<_T> intermediate;

                void Dispose();

                void ZeroOverwrite();
                void RandomOverwrite(Random::AnyRNG<uint64_t> rng);
                void RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng);

                std::pair<_T*, size_t> Run();

                GeneMLP<_T> Clone();
                void Randomize(_T Scalar, Random::AnyRNG<uint64_t> rng);
                void Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng);
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, Random::AnyRNG<uint64_t> rng);
                void Randomize(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng);
                GeneMLP<_T> Reproduce(_T Scalar, Random::AnyRNG<uint64_t> rng);
                GeneMLP<_T> Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng);
                GeneMLP<_T> Reproduce(_T Scalar_Base, _T Scalar_Intermediate, Random::AnyRNG<uint64_t> rng);
                GeneMLP<_T> Reproduce(_T Scalar_Base, _T Scalar_Intermediate, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> rng);
            private:
                std::pair<_T*, size_t> base;
            };
        }
    }
}