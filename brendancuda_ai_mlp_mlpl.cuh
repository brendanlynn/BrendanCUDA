#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <limits>
#include <random>
#include "brendancuda_devicecopy.cuh"
#include "brendancuda_ai.cuh"
#include "brendancuda_random_devicerandom.cuh"
#include "brendancuda_random_rngfunc.cuh"
#include <iostream>

namespace BrendanCUDA {
    namespace AI {
        namespace MLP {
            template <typename T>
            class MLPL {
                static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "T must be either 'float' or 'double'.");
            public:
                MLPL() = default;
                __host__ __device__ MLPL(size_t InputLength, size_t OutputLength);
                __host__ MLPL(size_t InputLength, size_t OutputLength, T* Weights, T* Bias, bool CopyFromHost);
                __device__ MLPL(size_t InputLength, size_t OutputLength, T* Weights, T* Bias);

                __host__ __device__ void Dispose();

                __host__ __device__ void ZeroOverwrite();
                __host__ __device__ void RandomOverwrite(Random::rngWState<uint64_t> rng);
                __host__ __device__ void RandomOverwrite(T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng);

                __host__ __device__ size_t InputLength() const;
                __host__ __device__ size_t OutputLength() const;
                __host__ __device__ T* Weights() const;
                __host__ __device__ T* Weight(size_t Index) const;
                __host__ __device__ T* Weight(size_t X, size_t Y) const;
                __host__ __device__ T* Bias() const;
                __host__ __device__ T* Bias(size_t Index) const;

                __host__ T* GetWeights(bool CopyToHost) const;
                __device__ T* GetWeights() const;
                __host__ __device__ T GetWeight(size_t Index);
                __host__ __device__ T GetWeight(size_t X, size_t Y);
                __host__ void SetWeights(T* Values, bool CopyFromHost);
                __device__ void SetWeights(T* Values);
                __host__ __device__ void SetWeight(T Value, size_t Index);
                __host__ __device__ void SetWeight(T Value, size_t X, size_t Y);
                __host__ T* GetBias(bool CopyToHost) const;
                __device__ T* GetBias() const;
                __host__ __device__ T GetBias(size_t Index) const;
                __host__ void SetBias(T* Values, bool CopyFromHost);
                __device__ void SetBias(T* Values);
                __host__ __device__ void SetBias(T Value, size_t Index);

                __host__ T* Run(T* Input) const;

                __host__ __device__ MLPL<T> Clone() const;
                __host__ __device__ void Randomize(T Scalar, Random::rngWState<uint64_t> rng);
                __host__ __device__ void Randomize(T Scalar, T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng);
                __host__ __device__ MLPL<T> Reproduce(T Scalar, Random::rngWState<uint64_t> rng) const;
                __host__ __device__ MLPL<T> Reproduce(T Scalar, T LowerBound, T UpperBound, Random::rngWState<uint64_t> rng) const;

                __host__ void Serialize(std::basic_ostream<char>& Stream);
                __host__ static MLPL<T> Deserialize(std::basic_istream<char>& Stream);
            private:
                size_t iptLen;
                size_t optLen;
                T* wgts;
                T* bias;
            };
        }
    }
}