#include "ai_evol_eval_output_impl_uniquevalues.h"
#include <exception>
#include <unordered_set>

template <typename _T>
float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues(void* Object, Evaluate_UniqueValues_SD<_T>& Settings) {
    Instance_V<_T*, _T*> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
    if (Settings.individual) {
        std::unordered_set<_T>* s = new std::unordered_set<_T>[Settings.outputCount];
        for (size_t i = 0; i < Settings.outputCount; ++i) {
            s[i] = std::unordered_set<_T>();
        }
        for (uint64_t i = 0; i < Settings.iterationCount; ++i) {
            _T* o = oi.IterateInstance(0);
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
        std::unordered_set<_T> s;
        for (uint64_t i = 0; i < Settings.iterationCount; ++i) {
            _T* o = oi.IterateInstance(0);
            for (size_t j = 0; j < Settings.outputCount; ++j) {
                s.insert(o[j]);
            }
            delete[] o;
        }
        oi.DestroyInstance();
        return s.size() / (float)(Settings.iterationCount * Settings.outputCount);
    }
}

template <typename _T>
float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues(void* Object, void* Settings) {
    return Evaluate_UniqueValues(Object, *(Evaluate_UniqueValues_SD<_T>*)Settings);
}

template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint8_t>(void*, Evaluate_UniqueValues_SD<uint8_t>&);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int8_t>(void*, Evaluate_UniqueValues_SD<int8_t>&);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint16_t>(void*, Evaluate_UniqueValues_SD<uint16_t>&);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int16_t>(void*, Evaluate_UniqueValues_SD<int16_t>&);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint32_t>(void*, Evaluate_UniqueValues_SD<uint32_t>&);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int32_t>(void*, Evaluate_UniqueValues_SD<int32_t>&);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint64_t>(void*, Evaluate_UniqueValues_SD<uint64_t>&);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int64_t>(void*, Evaluate_UniqueValues_SD<int64_t>&);

template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint8_t>(void*, void*);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int8_t>(void*, void*);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint16_t>(void*, void*);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int16_t>(void*, void*);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint32_t>(void*, void*);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int32_t>(void*, void*);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<uint64_t>(void*, void*);
template float brendancuda::ai::evolution::evaluation::output::Evaluate_UniqueValues<int64_t>(void*, void*);