#pragma once

#include "brendancuda_ai_mlp.cuh"
#include <vector>

namespace BrendanCUDA {
    namespace AI {
        namespace Genetics {
            template <typename T>
            class GeneMLP final {
            public:
                GeneMLP() = default;
                GeneMLP(std::pair<T*, size_t> Base, MLP::MLP<T> Intermediate);

                std::pair<T*, size_t> Base();
                MLP::MLP<T> Intermediate;

                void Dispose();

                void ZeroOverwrite();
                void RandomOverwrite(Random::rngWState<uint64_t> rng);
                void RandomOverwrite(T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng);

                std::pair<T*, size_t> Run();

                GeneMLP<T> Clone();
                void Randomize(T Scalar, Random::rngWState<uint64_t> rng);
                void Randomize(T Scalar, T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng);
                void Randomize(T Scalar_Base, T Scalar_Intermediate, Random::rngWState<uint64_t> rng);
                void Randomize(T Scalar_Base, T Scalar_Intermediate, T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng);
                GeneMLP<T> Reproduce(T Scalar, Random::rngWState<uint64_t> rng);
                GeneMLP<T> Reproduce(T Scalar, T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng);
                GeneMLP<T> Reproduce(T Scalar_Base, T Scalar_Intermediate, Random::rngWState<uint64_t> rng);
                GeneMLP<T> Reproduce(T Scalar_Base, T Scalar_Intermediate, T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng);
            private:
                std::pair<T*, size_t> base;
            };
        }
    }
}