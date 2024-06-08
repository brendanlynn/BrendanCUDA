#pragma once

#include "brendancuda_ai_evolution_evaluation_output.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                namespace Output {
                    namespace Implementations {
                        template <typename T>
                        float evalUniqueValues(void* Object, void* EvaluationSharedData);

                        template <typename T>
                        struct evalUniqueValues_sd final {
                            instanceFunctions_t<T> instanceFunctions;
                            uint64_t iterationCount;
                            size_t outputCount;
                            bool individual;
                            void* sd_ci;
                        };
                    }
                }
            }
        }
    }
}