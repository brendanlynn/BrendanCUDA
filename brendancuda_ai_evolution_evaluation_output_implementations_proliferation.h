#pragma once

#include "brendancuda_random_anyrng.cuh"
#include "brendancuda_ai_evolution_evaluation_output.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                namespace Output {
                    namespace Implementations {
                        template <typename T>
                        struct evalProliferation_sd final {
                            instanceFunctions_t<T> instanceFunctions;
                            uint64_t iterationsPerRound;
                            uint64_t roundCount;
                            size_t inputCount;
                            size_t outputCount;
                            bool individual;
                            T mask;
                            void* sd_ci;
                            BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        };
                        template <>
                        struct evalProliferation_sd<float> final {
                            instanceFunctions_t<float> instanceFunctions;
                            uint64_t iterationsPerRound;
                            uint64_t roundCount;
                            size_t inputCount;
                            size_t outputCount;
                            bool individual;
                            void* sd_ci;
                            BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        };
                        template <>
                        struct evalProliferation_sd<double> final {
                            instanceFunctions_t<double> instanceFunctions;
                            uint64_t iterationsPerRound;
                            uint64_t roundCount;
                            size_t inputCount;
                            size_t outputCount;
                            bool individual;
                            void* sd_ci;
                            BrendanCUDA::Random::AnyRNG<uint64_t> rng;
                        };

                        template <typename T>
                        float evalProliferation(void* Object, evalProliferation_sd<T>& Settings);
                    }
                }
            }
        }
    }
}