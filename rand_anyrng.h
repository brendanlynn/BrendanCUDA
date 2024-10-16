#pragma once

#include <concepts>
#include <cuda_runtime.h>
#include <limits>
#include <random>

namespace bcuda {
    namespace details {
        template <typename _TOutputType, typename _TRNG>
        static inline _TOutputType RunRNGFunc(void* RNG) {
            std::uniform_int_distribution<_TOutputType> dis(0);
            return dis(*(_TRNG*)RNG);
        }
        template <typename _TOutputType>
        using runRNGFunc_t = _TOutputType(*)(void*);
    }
    namespace rand {
        template <std::unsigned_integral _TOutputType>
        class AnyRNG {
        public:
            template <typename _TRNG>
            inline AnyRNG(_TRNG* RNG) {
                i_rng = RNG;
                r_rng = details::RunRNGFunc<_TOutputType, _TRNG>;
            }
            template <typename _TRNG>
            inline AnyRNG(_TRNG& RNG) {
                i_rng = &RNG;
                r_rng = details::RunRNGFunc<_TOutputType, _TRNG>;
            }
            inline _TOutputType operator()() {
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