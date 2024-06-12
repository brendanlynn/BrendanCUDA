#include "brendancuda_ai_evolution_evaluation_output_implementations_uniquevalues.h"
#include <unordered_set>
#include <exception>

template <typename T>
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues(void* Object, evalUniqueValues_sd<T>& Settings) {
    instance_v_t<T> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
    if (Settings.individual) {
        std::unordered_set<T>* s = new std::unordered_set<T>[Settings.outputCount];
        for (size_t i = 0; i < Settings.outputCount; ++i) {
            s[i] = std::unordered_set<T>();
        }
        for (uint64_t i = 0; i < Settings.iterationCount; ++i) {
            T* o = oi.IterateInstance(0);
            for (size_t j = 0; j < Settings.outputCount; ++j) {
                s[j].insert(o[j]);
            }
            delete[] o;
        }
        size_t c = 0;
        for (size_t i = 0; i < Settings.outputCount; ++i) {
            c += s[i].size();
        }
        delete[] s;
        oi.DestroyInstance();
        return c / (float)(Settings.iterationCount * Settings.outputCount);
    }
    else {
        std::unordered_set<T> s;
        for (uint64_t i = 0; i < Settings.iterationCount; ++i) {
            T* o = oi.IterateInstance(0);
            for (size_t j = 0; j < Settings.outputCount; ++j) {
                s.insert(o[j]);
            }
            delete[] o;
        }
        oi.DestroyInstance();
        return s.size() / (float)(Settings.iterationCount * Settings.outputCount);
    }
}

template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint8_t>(void*, evalUniqueValues_sd<uint8_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int8_t>(void*, evalUniqueValues_sd<int8_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint16_t>(void*, evalUniqueValues_sd<uint16_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int16_t>(void*, evalUniqueValues_sd<int16_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint32_t>(void*, evalUniqueValues_sd<uint32_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int32_t>(void*, evalUniqueValues_sd<int32_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<uint64_t>(void*, evalUniqueValues_sd<uint64_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalUniqueValues<int64_t>(void*, evalUniqueValues_sd<int64_t>&);