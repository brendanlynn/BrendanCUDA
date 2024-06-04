#pragma once

#include "brendancuda_random_rngfunc.cuh"
#include "brendancuda_ai_evolution_evaluation_output.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                namespace Output {
                    namespace Implementations {
                        template <typename T>
                        float evalProliferation(void* Object, void* EvaluationSharedData);

                        template <typename T>
                        struct evalProliferation_sd final {
                            instanceFunctions_t<T> instanceFunctions;
                            size_t iterationsPerRound;
                            size_t roundCount;
                            size_t inputCount;
                            size_t outputCount;
                            bool individual;
                            void* sd_ci;
                            BrendanCUDA::Random::rngWState<uint64_t> rng;
                        };
                    }
                }
            }
        }
    }
}