#pragma once

#include "ai_evol.h"
#include <algorithm>
#include <limits>

namespace bcuda {
    namespace ai {
        namespace evol {
            namespace eval {
                struct Evaluate_MultipleTimes_V_SD final {
                    size_t iterationCount;
                    void* internalEvaluationSharedData;
                };
                template <evaluationFunction_t _SingleEvaluationFunc, size_t _IterationCount>
                float Evaluate_MultipleTimes_AM_C(void* Object, void* EvaluationSharedData) {
                    constexpr float s = 1.f / _IterationCount;

                    float t = 0.f;
                    for (size_t i = 0; i < _IterationCount; ++i) {
                        t += _SingleEvaluationFunc(Object, EvaluationSharedData);
                    }

                    return t * s;
                }
                template <evaluationFunction_t _SingleEvaluationFunc>
                float Evaluate_MultipleTimes_AM_V(void* Object, Evaluate_MultipleTimes_V_SD& Settings) {
                    float t = 0.f;
                    for (size_t i = 0; i < Settings.iterationCount; ++i) {
                        t += _SingleEvaluationFunc(Object, Settings.internalEvaluationSharedData);
                    }

                    return t / Settings.iterationCount;
                }
                template <evaluationFunction_t _SingleEvaluationFunc, size_t _IterationCount>
                float Evaluate_MultipleTimes_Min_C(void* Object, void* EvaluationSharedData) {
                    float min = std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < _IterationCount; ++i) {
                        float v = _SingleEvaluationFunc(Object, EvaluationSharedData);
                        if (v < min) {
                            min = v;
                        }
                    }

                    return min;
                }
                template <evaluationFunction_t _SingleEvaluationFunc>
                float Evaluate_MultipleTimes_Min_V(void* Object, Evaluate_MultipleTimes_V_SD& Settings) {
                    float min = std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < Settings.iterationCount; ++i) {
                        float v = _SingleEvaluationFunc(Object, Settings.internalEvaluationSharedData);
                        if (v < min) {
                            min = v;
                        }
                    }

                    return min;
                }
                template <evaluationFunction_t _SingleEvaluationFunc, size_t _IterationCount>
                float Evaluate_MultipleTimes_Max_C(void* Object, void* EvaluationSharedData) {
                    float max = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < _IterationCount; ++i) {
                        float v = _SingleEvaluationFunc(Object, EvaluationSharedData);
                        if (v > max) {
                            max = v;
                        }
                    }

                    return max;
                }
                template <evaluationFunction_t _SingleEvaluationFunc>
                float Evaluate_MultipleTimes_Max_V(void* Object, Evaluate_MultipleTimes_V_SD& Settings) {
                    float max = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < Settings.iterationCount; ++i) {
                        float v = _SingleEvaluationFunc(Object, Settings.internalEvaluationSharedData);
                        if (v > max) {
                            max = v;
                        }
                    }

                    return max;
                }
                template <evaluationFunction_t _SingleEvaluationFunc, size_t _IterationCount>
                float Evaluate_MultipleTimes_Med_C(void* Object, void* EvaluationSharedData) {
                    float* arr = new float[_IterationCount];
                    for (size_t i = 0; i < _IterationCount; ++i) {
                        arr[i] = _SingleEvaluationFunc(Object, EvaluationSharedData);
                    }

                    std::sort(arr, arr + _IterationCount);

                    float r;
                    size_t h = _IterationCount >> 1;
                    if (_IterationCount & 1) {
                        r = arr[h];
                    }
                    else {
                        r = (arr[h - 1] + arr[h]) * .5f;
                    }

                    delete[] arr;

                    return r;
                }
                template <evaluationFunction_t _SingleEvaluationFunc>
                float Evaluate_MultipleTimes_Med_V(void* Object, Evaluate_MultipleTimes_V_SD& Settings) {
                    float* arr = new float[Settings.iterationCount];
                    for (size_t i = 0; i < Settings.iterationCount; ++i) {
                        arr[i] = _SingleEvaluationFunc(Object, Settings.internalEvaluationSharedData);
                    }

                    std::sort(arr, arr + Settings.iterationCount);

                    float r;
                    size_t h = Settings.iterationCount >> 1;
                    if (Settings.iterationCount & 1) {
                        r = arr[h];
                    }
                    else {
                        r = (arr[h - 1] + arr[h]) * .5f;
                    }

                    delete[] arr;

                    return r;
                }
            }
        }
    }
}