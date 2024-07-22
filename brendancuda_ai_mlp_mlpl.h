#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <limits>
#include <random>
#include "brendancuda_dcopy.cuh"
#include "brendancuda_ai.h"
#include "brendancuda_rand_anyrng.h"
#include <iostream>

namespace BrendanCUDA {
    namespace AI {
        namespace MLP {
            template <typename _T>
            class MLPL {
                static_assert(std::is_same<_T, float>::value || std::is_same<_T, double>::value, "_T must be either 'float' or 'double'.");
            public:
                MLPL() = default;
                __host__ __device__ MLPL(size_t InputLength, size_t OutputLength);
                __host__ MLPL(size_t InputLength, size_t OutputLength, _T* Weights, _T* Bias, bool CopyFromHost);
                __device__ MLPL(size_t InputLength, size_t OutputLength, _T* Weights, _T* Bias);

                __host__ __device__ void Dispose();

                __host__ __device__ void ZeroOverwrite();
                __host__ __device__ void RandomOverwrite(Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ void RandomOverwrite(_T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);

                __host__ __device__ size_t InputLength() const;
                __host__ __device__ size_t OutputLength() const;
                __host__ __device__ _T* Weights() const;
                __host__ __device__ _T* Weight(size_t Index) const;
                __host__ __device__ _T* Weight(size_t X, size_t Y) const;
                __host__ __device__ _T* Bias() const;
                __host__ __device__ _T* Bias(size_t Index) const;

                __host__ _T* GetWeights(bool CopyToHost) const;
                __device__ _T* GetWeights() const;
                __host__ __device__ _T GetWeight(size_t Index);
                __host__ __device__ _T GetWeight(size_t X, size_t Y);
                __host__ void SetWeights(_T* Values, bool CopyFromHost);
                __device__ void SetWeights(_T* Values);
                __host__ __device__ void SetWeight(_T Value, size_t Index);
                __host__ __device__ void SetWeight(_T Value, size_t X, size_t Y);
                __host__ _T* GetBias(bool CopyToHost) const;
                __device__ _T* GetBias() const;
                __host__ __device__ _T GetBias(size_t Index) const;
                __host__ void SetBias(_T* Values, bool CopyFromHost);
                __device__ void SetBias(_T* Values);
                __host__ __device__ void SetBias(_T Value, size_t Index);

                __host__ _T* Run(_T* Input) const;

                __host__ __device__ MLPL<_T> Clone() const;
                __host__ __device__ void Randomize(_T Scalar, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ void Randomize(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG);
                __host__ __device__ MLPL<_T> Reproduce(_T Scalar, Random::AnyRNG<uint64_t> RNG) const;
                __host__ __device__ MLPL<_T> Reproduce(_T Scalar, _T LowerBound, _T UpperBound, Random::AnyRNG<uint64_t> RNG) const;
            private:
                size_t iptLen;
                size_t optLen;
                _T* wgts;
                _T* bias;
            };
        }
    }
}