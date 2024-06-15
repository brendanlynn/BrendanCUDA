#pragma once

#include <random>
#include <limits>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace details {
        template <typename _TOutputType, typename _TRNG>
        __host__ __device__ TOutputType runRNGFunc(void* rng) {
            return (TOutputType)((*(TRNG*)rng)());
        }
        template <typename _TOutputType>
        using runRNGFunc_t = TOutputType(*)(void*);
    }
    namespace Random {
        template <typename _TOutputType>
        class AnyRNG final {
        public:
            template <typename _TRNG>
            __host__ __device__ AnyRNG(TRNG* rng) {
                i_rng = rng;
                r_rng = details::runRNGFunc<TOutputType, TRNG>;
            }
            __host__ __device__ TOutputType operator()() {
                return r_rng(i_rng);
            }
            __host__ __device__ static constexpr TOutputType min() {
                return std::numeric_limits<TOutputType>::min();
            }
            __host__ __device__ static constexpr TOutputType max() {
                return std::numeric_limits<TOutputType>::max();
            }
        private:
            void* i_rng;
            details::runRNGFunc_t<TOutputType> r_rng;
        };
    }
}