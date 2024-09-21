#pragma once

#include "ai_evol_eval_output.h"
#include "rand_anyrng.h"

namespace brendancuda {
    namespace ai {
        namespace evol {
            namespace evaluation {
                namespace output {
                    template <typename _T>
                    struct Evaluate_Proliferation_SD final {
                        InstanceFunctions<_T*, _T*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        _T mask;
                        void* sd_ci;
                        brendancuda::random::AnyRNG<uint64_t> rng;
                        __forceinline Evaluate_Proliferation_SD(brendancuda::random::AnyRNG<uint64_t> RNG);
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<float> final {
                        InstanceFunctions<float*, float*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        brendancuda::random::AnyRNG<uint64_t> rng;
                        __forceinline Evaluate_Proliferation_SD(brendancuda::random::AnyRNG<uint64_t> RNG);
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<double> final {
                        InstanceFunctions<double*, double*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        brendancuda::random::AnyRNG<uint64_t> rng;
                        __forceinline Evaluate_Proliferation_SD(brendancuda::random::AnyRNG<uint64_t> RNG);
                    };

                    template <typename _T>
                    float Evaluate_Proliferation(void* Object, Evaluate_Proliferation_SD<_T>& Settings);
                    template <typename _T>
                    float Evaluate_Proliferation(void* Object, void* Settings);
                }
            }
        }
    }
}

template <typename _T>
__forceinline brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation_SD<_T>::Evaluate_Proliferation_SD(brendancuda::random::AnyRNG<uint64_t> RNG)
    : rng(RNG),
    instanceFunctions() {
    iterationsPerRound = 0;
    roundCount = 0;
    inputCount = 0;
    outputCount = 0;
    mask = 0;
    sd_ci = 0;
}
__forceinline brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation_SD<float>::Evaluate_Proliferation_SD(brendancuda::random::AnyRNG<uint64_t> RNG)
    : rng(RNG),
    instanceFunctions() {
    iterationsPerRound = 0;
    roundCount = 0;
    inputCount = 0;
    outputCount = 0;
    sd_ci = 0;
}
__forceinline brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation_SD<double>::Evaluate_Proliferation_SD(brendancuda::random::AnyRNG<uint64_t> RNG)
    : rng(RNG),
    instanceFunctions() {
    iterationsPerRound = 0;
    roundCount = 0;
    inputCount = 0;
    outputCount = 0;
    sd_ci = 0;
}