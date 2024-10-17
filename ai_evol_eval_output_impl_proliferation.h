#pragma once

#include "ai_evol_eval_output.h"
#include "rand_anyrng.h"

namespace bcuda {
    namespace ai {
        namespace evol {
            namespace eval {
                namespace output {
                    template <typename _T>
                    struct Evaluate_Proliferation_SD final {
                        InstanceFunctions<_T*, _T*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        _T mask;
                        void* sd_ci;
                        bcuda::rand::AnyRNG<uint64_t> rng;
                        inline Evaluate_Proliferation_SD(bcuda::rand::AnyRNG<uint64_t> RNG)
                            : rng(RNG), instanceFunctions(), iterationsPerRound(0), roundCount(0), inputCount(0), outputCount(0), mask(0), sd_ci(0) { }
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<float> final {
                        InstanceFunctions<float*, float*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        bcuda::rand::AnyRNG<uint64_t> rng;
                        inline Evaluate_Proliferation_SD(bcuda::rand::AnyRNG<uint64_t> RNG)
                            : rng(RNG), instanceFunctions(), iterationsPerRound(0), roundCount(0), inputCount(0), outputCount(0), sd_ci(0) { }
                    };
                    template <>
                    struct Evaluate_Proliferation_SD<double> final {
                        InstanceFunctions<double*, double*> instanceFunctions;
                        uint64_t iterationsPerRound;
                        uint64_t roundCount;
                        size_t inputCount;
                        size_t outputCount;
                        void* sd_ci;
                        bcuda::rand::AnyRNG<uint64_t> rng;
                        inline Evaluate_Proliferation_SD(bcuda::rand::AnyRNG<uint64_t> RNG)
                            : rng(RNG), instanceFunctions(), iterationsPerRound(0), roundCount(0), inputCount(0), outputCount(0), sd_ci(0) { }
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