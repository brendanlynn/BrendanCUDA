#pragma once

#include <random>
#include <limits>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace details {
        template <typename _TOutputType, typename _TRNG>
        __host__ __device__ _TOutputType RunRNGFunc(void* rng) {
            return (_TOutputType)((*(_TRNG*)rng)());
        }
        template <typename _TOutputType>
        using runRNGFunc_t = _TOutputType(*)(void*);
    }
    namespace Random {
        template <typename _TOutputType>
        class AnyRNG final {
        public:
            template <typename _TRNG>
            __host__ __device__ AnyRNG(_TRNG* rng) {
                i_rng = rng;
                r_rng = details::RunRNGFunc<_TOutputType, _TRNG>;
            }
            __host__ __device__ _TOutputType operator()() {
                return r_rng(i_rng);
            }
            __host__ __device__ static constexpr _TOutputType min() {
                return std::numeric_limits<_TOutputType>::min();
            }
            __host__ __device__ static constexpr _TOutputType max() {
                return std::numeric_limits<_TOutputType>::max();
            }
        private:
            void* i_rng;
            details::runRNGFunc_t<_TOutputType> r_rng;
        };
    }
}