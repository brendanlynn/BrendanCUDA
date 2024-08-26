#include "ai_evol_evolver.h"
#include <algorithm>

BrendanCUDA::AI::Evolution::Evolver::Evolver(
    size_t ContestantCount,
    evaluationFunction_t EvaluationFunction,
    reproductionFunction_t ReproductionFunction,
    disposeFunction_t DisposeFunction
) : objs(ContestantCount) {
    evaluationFunction = EvaluationFunction;
    reproductionFunction = ReproductionFunction;
    disposeFunction = DisposeFunction;
}
BrendanCUDA::AI::Evolution::Evolver::Evolver(
    size_t ContestantCount,
    evaluationFunction_t EvaluationFunction,
    reproductionFunction_t ReproductionFunction,
    disposeFunction_t DisposeFunction,
    creationFunction_t CreationFunction,
    void* CreationSharedData
) : Evolver(ContestantCount, EvaluationFunction, ReproductionFunction, DisposeFunction) {
    InitAllNoDisposal(CreationFunction, CreationSharedData);
}
BrendanCUDA::ArrayV<void*> BrendanCUDA::AI::Evolution::Evolver::Objects() {
    return objs;
}
void BrendanCUDA::AI::Evolution::Evolver::InitAllNoDisposal(creationFunction_t CreationFunction, void* CreationSharedData) {
    std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

    for (size_t i = 0; i < objs.Size(); ++i) {
        objs[i] = CreationFunction(CreationSharedData);
    }
}
void BrendanCUDA::AI::Evolution::Evolver::InitAll(void* DisposeSharedData, creationFunction_t CreationFunction, void* CreationSharedData) {
    std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

    for (size_t i = 0; i < objs.Size(); ++i) {
        void*& v = objs[i];
        disposeFunction(v, DisposeSharedData);
        v = CreationFunction(CreationSharedData);
    }
}
BrendanCUDA::ArrayV<std::pair<float, size_t>> BrendanCUDA::AI::Evolution::Evolver::EvaluateAll(void* EvaluationSharedData) {
    std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

    ArrayV<std::pair<float, size_t>> scores(objs.Size());
    for (size_t i = 0; i < objs.Size(); ++i) {
        scores[i] = std::pair<float, size_t>(evaluationFunction(objs[i], EvaluationSharedData), i);
    }
    return scores;
}
void BrendanCUDA::AI::Evolution::SortEvaluations(ArrayV<std::pair<float, size_t>> Evaluations) {
    std::sort(Evaluations.Data(), Evaluations.Data() + Evaluations.Size(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
}
void BrendanCUDA::AI::Evolution::Evolver::ActOnSortedEvaluations(ArrayV<std::pair<float, size_t>> Evaluations, void* DisposeSharedData, void* ReproductionSharedData) {
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
BrendanCUDA::ArrayV<std::pair<float, size_t>> BrendanCUDA::AI::Evolution::Evolver::RunStep(void* EvaluationSharedData, void* ReproductionSharedData, void* DisposeSharedData) {
    auto eval = EvaluateAll(EvaluationSharedData);
    SortEvaluations(eval);
    ActOnSortedEvaluations(eval, DisposeSharedData, ReproductionSharedData);
    return eval;
}
void BrendanCUDA::AI::Evolution::Evolver::Dispose(void* DisposeSharedData) {
    for (size_t i = 0; i < objs.Size(); ++i) {
        void* v = objs[i];
        if (v) disposeFunction(v, DisposeSharedData);
    }
}