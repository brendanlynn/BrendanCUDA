#pragma once

#include "brendancuda_ai_evolution.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                template <evaluationFunction_t _SingleEvaluationFunc, size_t _IterationCount>
                float Evaluate_MultipleTimes_AM_C(void* Object, void* EvaluationSharedData) {
                    constexpr float s = 1.f / _IterationCount;

                    float t = 0.f;
                    for (size_t i = 0; i < _IterationCount; ++i) {
                        t += _SingleEvaluationFunc(Object, EvaluationSharedData);
                    }

                    return t / _IterationCount;
                }
                template <evaluationFunction_t _SingleEvaluationFunc>
                float Evaluate_MultipleTimes_AM_V(void* Object, void* EvaluationSharedData) {
                    Evaluate_MultipleTimes_AM_V_SD sd = *(Evaluate_MultipleTimes_AM_V_SD*)EvaluationSharedData;

                    float t = 0.f;
                    for (size_t i = 0; i < sd.iterationCount; ++i) {
                        t += _SingleEvaluationFunc(Object, sd.internalEvaluationSharedData);
                    }

                    return t / sd.iterationCount;
                }
                struct Evaluate_MultipleTimes_AM_V_SD final {
                    size_t iterationCount;
                    void* internalEvaluationSharedData;
                };
            }
        }
    }
}