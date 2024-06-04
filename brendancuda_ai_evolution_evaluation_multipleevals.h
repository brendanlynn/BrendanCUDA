#pragma once

#include "brendancuda_ai_evolution.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                template <evaluationFunction_t SingleEvaluationFunc, size_t IterationCount>
                float evalMultipleTimes_am_cc(void* Object, void* EvaluationSharedData) {
                    constexpr float s = 1.f / IterationCount;

                    float t = 0.f;
                    for (size_t i = 0; i < IterationCount; ++i) {
                        t += SingleEvaluationFunc(Object, EvaluationSharedData);
                    }

                    return t / IterationCount;
                }
                template <evaluationFunction_t SingleEvaluationFunc>
                float evalMultipleTimes_am_vc(void* Object, void* EvaluationSharedData) {
                    evalMultipleTimes_am_vc_sd sd = *(evalMultipleTimes_am_vc_sd*)EvaluationSharedData;
                    
                    float t = 0.f;
                    for (size_t i = 0; i < sd.iterationCount; ++i) {
                        t += SingleEvaluationFunc(Object, sd.internalEvaluationSharedData);
                    }

                    return t / sd.iterationCount;
                }
                struct evalMultipleTimes_am_vc_sd final {
                    size_t iterationCount;
                    void* internalEvaluationSharedData;
                };
            }
        }
    }
}