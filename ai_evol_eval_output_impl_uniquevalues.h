#pragma once

#include "ai_evol_eval_output.h"

namespace bcuda {
    namespace ai {
        namespace evol {
            namespace eval {
                namespace output {
                    template <typename _T>
                    struct Evaluate_UniqueValues_SD final {
                        InstanceFunctions<_T*, _T*> instanceFunctions;
                        uint64_t iterationCount;
                        size_t outputCount;
                        bool individual;
                        void* sd_ci;
                        inline constexpr Evaluate_UniqueValues_SD()
                            : instanceFunctions(), iterationCount(0), outputCount(0), individual(0), sd_ci(0) { }
                    };

                    template <typename _T>
                    float Evaluate_UniqueValues(void* Object, Evaluate_UniqueValues_SD<_T>& Settings);
                    template <typename _T>
                    float Evaluate_UniqueValues(void* Object, void* Settings);
                }
            }
        }
    }
}