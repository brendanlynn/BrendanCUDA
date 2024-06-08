#include "brendancuda_ai_evolution_evaluation_output_implementations_uniquevalues.h"
#include <unordered_set>
#include <exception>

template <typename T>
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues(void* Object, void* EvaluationSharedData) {
    evalUniqueValues_sd<T> sd = *(evalUniqueValues_sd<T>*)EvaluationSharedData;
    instance_v_t<T> oi(sd.instanceFunctions, Object, sd.sd_ci);
    if (sd.individual) {
        std::unordered_set<T>* s = new std::unordered_set<T>[sd.outputCount];
        for (size_t i = 0; i < sd.outputCount; ++i) {
            s[i] = std::unordered_set<T>();
        }
        for (uint64_t i = 0; i < sd.iterationCount; ++i) {
            T* o = oi.IterateInstance(0);
            for (size_t j = 0; j < sd.outputCount; ++j) {
                s[j].insert(o[j]);
            }
            delete[] o;
        }
        size_t c = 0;
        for (size_t i = 0; i < sd.outputCount; ++i) {
            c += s[i].size();
        }
        delete[] s;
        oi.DestroyInstance();
        return c / (float)(sd.iterationCount * sd.outputCount);
    }
    else {
        std::unordered_set<T> s;
        for (uint64_t i = 0; i < sd.iterationCount; ++i) {
            T* o = oi.IterateInstance(0);
            for (size_t j = 0; j < sd.outputCount; ++j) {
                s.insert(o[j]);
            }
            delete[] o;
        }
        oi.DestroyInstance();
        return s.size() / (float)(sd.iterationCount * sd.outputCount);
    }
}

template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint8_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int8_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint16_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int16_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint32_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int32_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint64_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int64_t>(void*, void*);