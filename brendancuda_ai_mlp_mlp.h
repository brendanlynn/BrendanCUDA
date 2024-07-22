#pragma once

#include "brendancuda_ai_mlp_mlpl.h"
#include "brendancuda_rand_anyrng.h"
#include "brendancuda_arrays.h"

namespace BrendanCUDA {
    namespace details {
        template <typename _T>
        __host__ __device__ void RunActivationFunctionOnArray(Span<_T> Array, AI::activationFunction_t<_T> ActivationFunction);
    }
    namespace AI {
        namespace MLP {
            template <typename _T>
            class MLP final {
                static_assert(std::is_same<_T, float>::value || std::is_same<_T, double>::value, "_T must be either 'float' or 'double'.");
            public:
                MLP() = default;
                __host__ __device__ MLP(size_t Length, activationFunction_t<_T> ActivationFunction);
                __host__ MLP(size_t Length, activationFunction_t<_T> ActivationFunction, MLPL<_T>* Layers, bool CopyFromHost);
                __device__ MLP(size_t Length, activationFunction_t<_T> ActivationFunction, MLPL<_T>* Layers);

                __host__ __device__ MLPL<_T>* Layers() const;
                __host__ __device__ size_t LayerCount() const;
                __host__ __device__ activationFunction_t<_T> ActivationFunction() const;

                __host__ __device__ void Dispose();

                __host__ __device__ void ZeroOverwrite();
                __host__ __device__ void RandomOverwrite(Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ void RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);

                __host__ __device__ MLPL<_T>* Layer(size_t LayerIndex) const;
                __host__ MLPL<_T>* GetLayers(bool CopyToHost) const;
                __device__ MLPL<_T>* GetLayers() const;
                __host__ void SetLayers(MLPL<_T>* Layers, bool CopyFromHost);
                __device__ void SetLayers(MLPL<_T>* Layers);
                __host__ __device__ MLPL<_T> GetLayer(size_t LayerIndex) const;
                __host__ __device__ void SetLayer(size_t LayerIndex, MLPL<_T> Layer);

                __host__ __device__ size_t InputLength();
                __host__ __device__ size_t OutputLength();

                __host__ _T* Run(_T* Input) const;

                __host__ __device__ MLP<_T> Clone() const;
                __host__ __device__ void Randomize(_T Scalar, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ void Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLP<_T> Reproduce(_T Scalar, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ MLP<_T> Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream) const;
                __host__ static MLP<_T> Deserialize(std::basic_istream<char>& Stream, activationFunction_t<_T> ActivationFunction);
            private:
                size_t len;
                MLPL<_T>* lyrs;
                activationFunction_t<_T> actnFunc;
            };
        }
    }
}