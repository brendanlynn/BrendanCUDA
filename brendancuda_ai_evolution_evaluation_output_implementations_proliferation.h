#pragma once

#include "brendancuda_random_anyrng.cuh"
#include "brendancuda_ai_evolution_evaluation_output.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                namespace Output {
                    template <typename _T>
                    struct Evaluate_Proliferation_SD final {
                        InstanceFunctions<_T> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        _T mask;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        inline Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG)
                            : rng(RNG) {
                            iterationsPerRound = 0;
                            roundCount = 0;
                            inputCount = 0;
                            outputCount = 0;
                            mask = 0;
                            sd_ci = 0;
                        }
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<float> final {
                        InstanceFunctions<float> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        inline Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG)
                            : rng(RNG) {
                            iterationsPerRound = 0;
                            roundCount = 0;
                            inputCount = 0;
                            outputCount = 0;
                            sd_ci = 0;
                        }
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<double> final {
                        InstanceFunctions<double> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        inline Evaluate_Proliferation_SD(BrendanCUDA::Random::AnyRNG<uint64_t> RNG)
                            : rng(RNG) {
                            iterationsPerRound = 0;
                            roundCount = 0;
                            inputCount = 0;
                            outputCount = 0;
                            sd_ci = 0;
                        }
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