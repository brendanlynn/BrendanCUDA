#pragma once

#include <cstdint>
#include <vector>
#include <random>

#include "brendancuda_ai_evolution.h"

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            //A class to reduce the boilerplate code of implementing an evolutionary algorithm.
            class Evolver final {
            public:
                //Creates an instance of the BrendanCUDA::AI::Evolution::Evolver class.
                Evolver(
                    size_t ContestantCount,
                    evaluationFunction_t EvaluationFunction,
                    reproductionFunction_t ReproductionFunction,
                    disposeFunction_t DisposeFunction
                );
                //Creates an instance of the BrendanCUDA::AI::Evolution::Evolver class.
                Evolver(
                    size_t ContestantCount,
                    evaluationFunction_t EvaluationFunction,
                    reproductionFunction_t ReproductionFunction,
                    disposeFunction_t DisposeFunction,
                    creationFunction_t CreationFunction,
                    void* CreationSharedData
                );
                //An array of all the objects in the evolutionary system.
                std::pair<void**, size_t> Objects();
                //Initializes the array of objects without attempting to dispose of any existing objects, if valid.
                void InitAllNoDisposal(creationFunction_t CreationFunction, void* CreationSharedData);
                //Disposes of previous objects and then reinitializes the array to be of all new objects.
                void InitAll(void* DisposeSharedData, creationFunction_t CreationFunction, void* CreationSharedData);
                //Disposes of the class, including the objects current in the evolutionary system.
                void Dispose(void* DisposeSharedData);
                //Evaluates the objects currently present, and returns their respective scores in a format ready for sorting.
                std::pair<std::pair<float, size_t>*, size_t> EvaluateAll(void* EvaluationSharedData);
                //Disposes of below-average performers, and replaces them with the offspring of the above-average performers.
                void ActOnSortedEvaluations(std::pair<std::pair<float, size_t>*, size_t> Evaluations, void* DisposeSharedData, void* ReproductionSharedData);
                //Evaluates and undergoes the reproduction of all the objects present.
                std::pair<std::pair<float, size_t>*, size_t> RunStep(void* EvaluationSharedData, void* ReproductionSharedData, void* DisposeSharedData);
                //The evaluation function of the evolutionary system, utilized in EvaluateAll and, by proxy, RunStep.
                evaluationFunction_t evaluationFunction;
                //The reproduction function of the evolutionary system, utilized in ActOnSortedEvaluations and, by proxy, RunStep.
                reproductionFunction_t reproductionFunction;
                //The disposal function of the evolutionary system, utilized in InitAll, ActOnSortedEvaluations, and, by proxy, RunStep.
                disposeFunction_t disposeFunction;
            private:
                std::pair<void**, size_t> objs;
            };
            //Sorts the evaluations of evaluated objects evaluated in the evaluation process of an evolutionary system.
            void SortEvaluations(std::pair<std::pair<float, size_t>*, size_t> Evaluations);
        }
    }
}
