#pragma once

#include "brendancuda_ai_mlp_mlpl.cuh"
#include "brendancuda_devicecopy.cuh"
#include "brendancuda_random_anyrng.cuh"

namespace BrendanCUDA {
    namespace AI {
        namespace MLP {
            template <typename T>
            class MLP final {
                static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "T must be either 'float' or 'double'.");
            public:
                MLP() = default;
                __host__ __device__ MLP(size_t Length, activationFunction_t<T> ActivationFunction);
                __host__ MLP(size_t Length, activationFunction_t<T> ActivationFunction, MLPL<T>* Layers, bool CopyFromHost);
                __device__ MLP(size_t Length, activationFunction_t<T> ActivationFunction, MLPL<T>* Layers);

                __host__ __device__ MLPL<T>* Layers() const;
                __host__ __device__ size_t LayerCount() const;
                __host__ __device__ activationFunction_t<T> ActivationFunction() const;

                __host__ __device__ void Dispose();

                __host__ __device__ void ZeroOverwrite();
                __host__ __device__ void RandomOverwrite(Random::AnyRNG<uint64_t> rng);
                __host__ __device__ void RandomOverwrite(T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng);

                __host__ __device__ MLPL<T>* Layer(size_t LayerIndex) const;
                __host__ MLPL<T>* GetLayers(bool CopyToHost) const;
                __device__ MLPL<T>* GetLayers() const;
                __host__ void SetLayers(MLPL<T>* Layers, bool CopyFromHost);
                __device__ void SetLayers(MLPL<T>* Layers);
                __host__ __device__ MLPL<T> GetLayer(size_t LayerIndex) const;
                __host__ __device__ void SetLayer(size_t LayerIndex, MLPL<T> Layer);

                __host__ __device__ size_t InputLength();
                __host__ __device__ size_t OutputLength();

                __host__ std::pair<T*, size_t> Run(T* Input);

                __host__ __device__ MLP<T> Clone() const;
                __host__ __device__ void Randomize(T Scalar, Random::AnyRNG<uint64_t> rng);
                __host__ __device__ void Randomize(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng);
                __host__ __device__ MLP<T> Reproduce(T Scalar, Random::AnyRNG<uint64_t> rng) const;
                __host__ __device__ MLP<T> Reproduce(T Scalar, T LowerBound, T UpperBound, Random::AnyRNG<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream);
                __host__ static MLP<T> Deserialize(std::basic_istream<char>& Stream, activationFunction_t<T> ActivationFunction);
            private:
                size_t len;
                MLPL<T>* lyrs;
                activationFunction_t<T> actnFunc;

                __host__ __device__ void RunActivationFunctionOnArray(T* Array, size_t Length);
            };
        }
    }
}