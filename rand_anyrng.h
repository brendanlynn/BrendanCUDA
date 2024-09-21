#pragma once

#include <concepts>
#include <cuda_runtime.h>
#include <limits>
#include <random>

namespace brendancuda {
    namespace details {
        template <typename _TOutputType, typename _TRNG>
        _TOutputType RunRNGFunc(void* RNG) {
            std::uniform_int_distribution<_TOutputType> dis(0);
            return dis(*(_TRNG*)RNG);
        }
        template <typename _TOutputType>
        using runRNGFunc_t = _TOutputType(*)(void*);
    }
    namespace Random {
        template <std::unsigned_integral _TOutputType>
        class AnyRNG {
        public:
            template <typename _TRNG>
            AnyRNG(_TRNG* RNG) {
                i_rng = RNG;
                r_rng = details::RunRNGFunc<_TOutputType, _TRNG>;
            }
            template <typename _TRNG>
            AnyRNG(_TRNG& RNG) {
                i_rng = &RNG;
                r_rng = details::RunRNGFunc<_TOutputType, _TRNG>;
            }
            _TOutputType operator()() {
                return r_rng(i_rng);
            }
            static constexpr _TOutputType min() {
                return std::numeric_limits<_TOutputType>::min();
            }
            static constexpr _TOutputType max() {
                return std::numeric_limits<_TOutputType>::max();
            }
            using result_type = _TOutputType;
        private:
            void* i_rng;
            details::runRNGFunc_t<_TOutputType> r_rng;
        };
    }
}