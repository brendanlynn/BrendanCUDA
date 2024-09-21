#include "ai_evol_eval_output_impl_proliferation.h"
#include "binary_basic.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

template <>
float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<float>(void* Object, Evaluate_Proliferation_SD<float>& Settings) {
    constexpr float rndSclr = 2.f / (float)std::numeric_limits<uint64_t>::max();
    if (!Settings.inputCount) {
        throw std::runtime_error("'Settings.inputCount' cannot be zero.");
    }
    Instance_V<float*, float*> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
    float t = 0.f;
    for (size_t i = 0; i < Settings.roundCount; ++i) {
        float* vs = new float[Settings.inputCount];
        float rndVal = Settings.rng() * rndSclr - 1.f;
        for (size_t j = 0; j < Settings.inputCount; ++j) {
            vs[j] = rndVal;
        }
        for (size_t j = 0; j < Settings.iterationsPerRound; ++j) {
            float* r = oi.IterateInstance(vs);
            float dev_t = 0.f;
            for (size_t k = 0; k < Settings.outputCount; ++k) {
                float iv = r[k] - rndVal;
                dev_t += iv * iv;
            }
            delete[] r;
            t += std::sqrt(dev_t);
        }
        delete[] vs;
    }
    return t;
}

template<>
float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<double>(void* Object, Evaluate_Proliferation_SD<double>& Settings) {
    constexpr double rndSclr = 2. / (double)std::numeric_limits<uint64_t>::max();

    if (!Settings.inputCount) {
        throw std::runtime_error("'Settings.inputCount' cannot be zero.");
    }
    Instance_V<double*, double*> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
    double t = 0.;
    for (size_t i = 0; i < Settings.roundCount; ++i) {
        double* vs = new double[Settings.inputCount];
        double rndVal = Settings.rng() * rndSclr - 1.;
        for (size_t j = 0; j < Settings.inputCount; ++j) {
            vs[j] = rndVal;
        }
        for (size_t j = 0; j < Settings.iterationsPerRound; ++j) {
            double* r = oi.IterateInstance(vs);
            double dev_t = 0.;
            for (size_t k = 0; k < Settings.outputCount; ++k) {
                double iv = r[k] - rndVal;
                dev_t += iv * iv;
            }
            delete[] r;
            t += std::sqrt(dev_t);
        }
        delete[] vs;
    }
    return (float)t;
}

template <typename _T>
float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation(void* Object, Evaluate_Proliferation_SD<_T>& Settings) {
    if (!Settings.inputCount) {
        throw std::runtime_error("'Settings.inputCount' cannot be zero.");
    }
    Instance_V<_T*, _T*> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
    uint64_t t = 0ui64;
    for (size_t i = 0; i < Settings.roundCount; ++i) {
        _T* vs = new _T[Settings.inputCount];
        _T rndVal = (_T)Settings.rng();
        for (size_t j = 0; j < Settings.inputCount; ++j) {
            vs[j] = rndVal;
        }
        for (size_t j = 0; j < Settings.iterationsPerRound; ++j) {
            _T* r = oi.IterateInstance(vs);
            for (size_t k = 0; k < Settings.outputCount; ++k) {
                _T iv = Settings.mask & ~(r[k] ^ rndVal);
                if constexpr (std::is_same<int8_t, _T>::value || std::is_same<uint8_t, _T>::value) {
                    t += std::popcount((uint8_t)iv);
                }
                else if constexpr (std::is_same<int16_t, _T>::value || std::is_same<uint16_t, _T>::value) {
                    t += std::popcount((uint16_t)iv);
                }
                else if constexpr (std::is_same<int32_t, _T>::value || std::is_same<uint32_t, _T>::value) {
                    t += std::popcount((uint32_t)iv);
                }
                else if constexpr (std::is_same<int64_t, _T>::value || std::is_same<uint64_t, _T>::value) {
                    t += std::popcount((uint64_t)iv);
                }
                else {
                    throw std::exception();
                }
            }
            delete[] r;
        }
        delete[] vs;
    }
    _T mc;
    if constexpr (std::is_same<int8_t, _T>::value || std::is_same<uint8_t, _T>::value) {
        mc = std::popcount((uint8_t)Settings.mask);
    }
    else if constexpr (std::is_same<int16_t, _T>::value || std::is_same<uint16_t, _T>::value) {
        mc = std::popcount((uint16_t)Settings.mask);
    }
    else if constexpr (std::is_same<int32_t, _T>::value || std::is_same<uint32_t, _T>::value) {
        mc = std::popcount((uint32_t)Settings.mask);
    }
    else if constexpr (std::is_same<int64_t, _T>::value || std::is_same<uint64_t, _T>::value) {
        mc = std::popcount((uint64_t)Settings.mask);
    }
    else {
        throw std::exception();
    }
    return (float)t / (float)((uint64_t)mc * Settings.iterationsPerRound * Settings.roundCount * Settings.outputCount);
}

template <typename _T>
float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation(void* Object, void* Settings) {
    return Evaluate_Proliferation<_T>(Object, *(Evaluate_Proliferation_SD<_T>*)Settings);
}

template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint8_t>(void*, Evaluate_Proliferation_SD<uint8_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int8_t>(void*, Evaluate_Proliferation_SD<int8_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint16_t>(void*, Evaluate_Proliferation_SD<uint16_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int16_t>(void*, Evaluate_Proliferation_SD<int16_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint32_t>(void*, Evaluate_Proliferation_SD<uint32_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int32_t>(void*, Evaluate_Proliferation_SD<int32_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint64_t>(void*, Evaluate_Proliferation_SD<uint64_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int64_t>(void*, Evaluate_Proliferation_SD<int64_t>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<float>(void*, Evaluate_Proliferation_SD<float>&);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<double>(void*, Evaluate_Proliferation_SD<double>&);

template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint8_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int8_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint16_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int16_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint32_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int32_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<uint64_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<int64_t>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<float>(void*, void*);
template float brendancuda::ai::evol::evaluation::output::Evaluate_Proliferation<double>(void*, void*);