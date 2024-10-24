#include "ai_evol_evolver.h"
#include <algorithm>

namespace bcuda {
    namespace ai {
        namespace evol {
            Evolver::Evolver(size_t ContestantCount, evaluationFunction_t EvaluationFunction, reproductionFunction_t ReproductionFunction, disposeFunction_t DisposeFunction)
                : objs(ContestantCount), evaluationFunction(EvaluationFunction), reproductionFunction(ReproductionFunction), disposeFunction(DisposeFunction) { }
            Evolver::Evolver(size_t ContestantCount, evaluationFunction_t EvaluationFunction, reproductionFunction_t ReproductionFunction, disposeFunction_t DisposeFunction, creationFunction_t CreationFunction, void* CreationSharedData)
                : Evolver(ContestantCount, EvaluationFunction, ReproductionFunction, DisposeFunction) {
                InitAllNoDisposal(CreationFunction, CreationSharedData);
            }
            bcuda::ArrayV<void*> Evolver::Objects() {
                return objs;
            }
            void Evolver::InitAllNoDisposal(creationFunction_t CreationFunction, void* CreationSharedData) {
                std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

                for (size_t i = 0; i < objs.Size(); ++i) {
                    objs[i] = CreationFunction(CreationSharedData);
                }
            }
            void Evolver::InitAll(void* DisposeSharedData, creationFunction_t CreationFunction, void* CreationSharedData) {
                std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

                for (size_t i = 0; i < objs.Size(); ++i) {
                    void*& v = objs[i];
                    disposeFunction(v, DisposeSharedData);
                    v = CreationFunction(CreationSharedData);
                }
            }
            bcuda::ArrayV<std::pair<float, size_t>> Evolver::EvaluateAll(void* EvaluationSharedData) {
                std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

                ArrayV<std::pair<float, size_t>> scores(objs.Size());
                for (size_t i = 0; i < objs.Size(); ++i) {
                    scores[i] = std::pair<float, size_t>(evaluationFunction(objs[i], EvaluationSharedData), i);
                }
                return scores;
            }
            void SortEvaluations(ArrayV<std::pair<float, size_t>> Evaluations) {
                std::sort(Evaluations.Data(), Evaluations.Data() + Evaluations.Size(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                    });
            }
            void Evolver::ActOnSortedEvaluations(ArrayV<std::pair<float, size_t>> Evaluations, void* DisposeSharedData, void* ReproductionSharedData) {
                std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

                if (Evaluations.Size() != objs.Size())
                    throw new std::exception("Input array 'Evaluations' must be the same size as field 'Objects'.");

                size_t s1_2 = objs.Size() >> 1;

                size_t i;
                size_t j = objs.Size() - s1_2;
                for (i = 0; i < s1_2; ++i) {
                    void*& iVP(objs[Evaluations[i].second]);
                    void*& jVP(objs[Evaluations[j].second]);
                    void* n = reproductionFunction(jVP, ReproductionSharedData);
                    disposeFunction(iVP, DisposeSharedData);
                    iVP = n;

                    ++j;
                }
            }
            bcuda::ArrayV<std::pair<float, size_t>> Evolver::RunStep(void* EvaluationSharedData, void* ReproductionSharedData, void* DisposeSharedData) {
                auto eval = EvaluateAll(EvaluationSharedData);
                SortEvaluations(eval);
                ActOnSortedEvaluations(eval, DisposeSharedData, ReproductionSharedData);
                return eval;
            }
            void Evolver::Dispose(void* DisposeSharedData) {
                for (size_t i = 0; i < objs.Size(); ++i) {
                    void* v = objs[i];
                    if (v) disposeFunction(v, DisposeSharedData);
                }
            }
        }
    }
}