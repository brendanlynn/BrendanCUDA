#pragma once

#include <cstdint>
#include <vector>
#include <random>

#include "brendancuda_ai_evolution.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            class Evolver final {
            public:
                Evolver(
                    size_t ContestantCount,
                    evaluationFunction_t EvaluationFunction,
                    reproductionFunction_t ReproductionFunction,
                    disposeFunction_t DisposeFunction
                );
                Evolver(
                    size_t ContestantCount,
                    evaluationFunction_t EvaluationFunction,
                    reproductionFunction_t ReproductionFunction,
                    disposeFunction_t DisposeFunction,
                    creationFunction_t CreationFunction,
                    void* CreationSharedData
                );
                std::pair<void**, size_t> Objects();
                void InitAllNoDisposal(creationFunction_t CreationFunction, void* CreationSharedData);
                void InitAll(void* DisposeSharedData, creationFunction_t CreationFunction, void* CreationSharedData);
                void Dispose(void* DisposeSharedData);
                std::pair<std::pair<float, size_t>*, size_t> EvaluateAll(void* EvaluationSharedData);
                void ActOnSortedEvaluations(std::pair<std::pair<float, size_t>*, size_t> Evaluations, void* DisposeSharedData, void* ReproductionSharedData);
                std::pair<std::pair<float, size_t>*, size_t> RunStep(void* EvaluationSharedData, void* ReproductionSharedData, void* DisposeSharedData);
                evaluationFunction_t evaluationFunction;
                reproductionFunction_t reproductionFunction;
                disposeFunction_t disposeFunction;
            private:
                std::pair<void**, size_t> objs;
            };
            void SortEvaluations(std::pair<std::pair<float, size_t>*, size_t> Evaluations);
        }
    }
}
