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
                        bool individual;
                        _T mask;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<float> final {
                        InstanceFunctions<float> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        bool individual;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<double> final {
                        InstanceFunctions<double> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        bool individual;
                        void* sd_ci;
                        BrendanCUDA::Random::AnyRNG<uint64_t> rng;
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