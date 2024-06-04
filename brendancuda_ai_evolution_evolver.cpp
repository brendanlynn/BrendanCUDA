#include "brendancuda_ai_evolution_evolver.h"
#include <algorithm>

BrendanCUDA::AI::Evolution::Evolver::Evolver(
    size_t ContestantCount,
    evaluationFunction_t EvaluationFunction,
    reproductionFunction_t ReproductionFunction,
    disposeFunction_t DisposeFunction
) {
    objs = std::pair<void**, size_t>(new void*[ContestantCount], ContestantCount);
    this->EvaluationFunction = EvaluationFunction;
    this->ReproductionFunction = ReproductionFunction;
    this->DisposeFunction = DisposeFunction;
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
std::pair<void**, size_t> BrendanCUDA::AI::Evolution::Evolver::Objects() {
    return objs;
}
void BrendanCUDA::AI::Evolution::Evolver::InitAllNoDisposal(creationFunction_t CreationFunction, void* CreationSharedData) {
    std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

    for (size_t i = 0; i < objs.second; ++i) {
        objs.first[i] = CreationFunction(CreationSharedData);
    }
}
void BrendanCUDA::AI::Evolution::Evolver::InitAll(void* DisposeSharedData, creationFunction_t CreationFunction, void* CreationSharedData) {
    std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

    for (size_t i = 0; i < objs.second; ++i) {
        void*& v = objs.first[i];
        DisposeFunction(v, DisposeSharedData);
        v = CreationFunction(CreationSharedData);
    }
}
std::pair<std::pair<float, size_t>*, size_t> BrendanCUDA::AI::Evolution::Evolver::EvaluateAll(void* EvaluationSharedData) {
    std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());

    std::pair<std::pair<float, size_t>*, size_t> scores = std::pair<std::pair<float, size_t>*, size_t>(new std::pair<float, size_t>[objs.second], objs.second);
    for (size_t i = 0; i < objs.second; ++i) {
        scores.first[i] = std::pair<float, size_t>(EvaluationFunction(objs.first[i], EvaluationSharedData), i);
    }
    return scores;
}
void BrendanCUDA::AI::Evolution::SortEvaluations(std::pair<std::pair<float, size_t>*, size_t> Evaluations) {
    std::sort(Evaluations.first, Evaluations.first + Evaluations.second, [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
}
void BrendanCUDA::AI::Evolution::Evolver::ActOnSortedEvaluations(std::pair<std::pair<float, size_t>*, size_t> Evaluations, void* DisposeSharedData, void* ReproductionSharedData) {
    std::uniform_int_distribution<uint64_t> dis(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());
    
    if (Evaluations.second != objs.second)
        throw new std::exception("Input array 'Evaluations' must be the same size as field 'Objects'.");

    size_t s1_2 = objs.second >> 1;

    size_t i;
    size_t j = objs.second - s1_2;
    for (i = 0; i < s1_2; ++i) {
        void*& iVP(objs.first[Evaluations.first[i].second]);
        void*& jVP(objs.first[Evaluations.first[j].second]);
        void* n = ReproductionFunction(jVP, ReproductionSharedData);
        DisposeFunction(iVP, DisposeSharedData);
        iVP = n;

        ++j;
    }
}
std::pair<std::pair<float, size_t>*, size_t> BrendanCUDA::AI::Evolution::Evolver::RunStep(void* EvaluationSharedData, void* ReproductionSharedData, void* DisposeSharedData) {
    auto eval = EvaluateAll(EvaluationSharedData);
    SortEvaluations(eval);
    ActOnSortedEvaluations(eval, DisposeSharedData, ReproductionSharedData);
    return eval;
}
void BrendanCUDA::AI::Evolution::Evolver::Dispose(void* DisposeSharedData) {
    for (size_t i = 0; i < objs.second; ++i) {
        void* v = objs.first[i];
        if (v != 0) {
            DisposeFunction(v, DisposeSharedData);
        }
    }
    delete objs.first;
}