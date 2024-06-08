#pragma once

#include <random>
#include <limits>

namespace BrendanCUDA {
    namespace Random {
        namespace {
            template <typename TOutputType, typename TRNG>
            TOutputType runRNGFunc(void* rng) {
                return (TOutputType)((*(TRNG*)rng)());
            }
            template <typename TOutputType>
            using runRNGFunc_t = TOutputType(*)(void*);
        }
        template <typename TOutputType>
        class AnyRNG final {
        public:
            template <typename TRNG>
            AnyRNG(TRNG* rng) {
                i_rng = rng;
                r_rng = runRNGFunc<TOutputType, TRNG>;
            }
            TOutputType operator()() {
                return r_rng(i_rng);
            }
            static constexpr TOutputType min() {
                return std::numeric_limits<TOutputType>::min();
            }
            static constexpr TOutputType max() {
                return std::numeric_limits<TOutputType>::max();
            }
        private:
            void* i_rng;
            runRNGFunc_t<TOutputType> r_rng;
        };
    }
}