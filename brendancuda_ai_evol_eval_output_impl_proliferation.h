#pragma once

#include "brendancuda_ai_evol_eval_output.h"
#include "brendancuda_rand_anyrng.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                namespace Output {
                    template <typename _T>
                    struct Evaluate_Proliferation_SD final {
                        InstanceFunctions<_T*, _T*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        _T mask;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        __forceinline Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<float> final {
                        InstanceFunctions<float*, float*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        __forceinline Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<double> final {
                        InstanceFunctions<double*, double*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        __forceinline Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG);
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
__forceinline BrendanCUDA::AI::Evolution::Evaluation::Output::Evaluate_Proliferation_SD<_T>::Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG)
    : rng(RNG),
    instanceFunctions() {
    iterationsPerRound = 0;
    roundCount = 0;
    inputCount = 0;
    outputCount = 0;
    mask = 0;
    sd_ci = 0;
}
__forceinline BrendanCUDA::AI::Evolution::Evaluation::Output::Evaluate_Proliferation_SD<float>::Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG)
    : rng(RNG),
    instanceFunctions() {
    iterationsPerRound = 0;
    roundCount = 0;
    inputCount = 0;
    outputCount = 0;
    sd_ci = 0;
}
__forceinline BrendanCUDA::AI::Evolution::Evaluation::Output::Evaluate_Proliferation_SD<double>::Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG)
    : rng(RNG),
    instanceFunctions() {
    iterationsPerRound = 0;
    roundCount = 0;
    inputCount = 0;
    outputCount = 0;
    sd_ci = 0;
}