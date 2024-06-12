#include "brendancuda_ai_evolution_evaluation_output_implementations_proliferation.h"
#include "brendancuda_binary_basic.cuh"
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <cmath>

template <>
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<float>(void* Object, evalProliferation_sd<float>& Settings) {
    constexpr float rndSclr = 2.f / (float)std::numeric_limits<uint64_t>::max();
    if (!Settings.inputCount) {
        throw std::runtime_error("'Settings.inputCount' cannot be zero.");
    }
    instance_v_t<float> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
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
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<double>(void* Object, evalProliferation_sd<double>& Settings) {
    constexpr double rndSclr = 2. / (double)std::numeric_limits<uint64_t>::max();

    if (!Settings.inputCount) {
        throw std::runtime_error("'Settings.inputCount' cannot be zero.");
    }
    instance_v_t<double> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
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

template <typename T>
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation(void* Object, evalProliferation_sd<T>& Settings) {
    if (!Settings.inputCount) {
        throw std::runtime_error("'Settings.inputCount' cannot be zero.");
    }
    instance_v_t<T> oi(Settings.instanceFunctions, Object, Settings.sd_ci);
    uint64_t t = 0ui64;
    for (size_t i = 0; i < Settings.roundCount; ++i) {
        T* vs = new T[Settings.inputCount];
        T rndVal = (T)Settings.rng();
        for (size_t j = 0; j < Settings.inputCount; ++j) {
            vs[j] = rndVal;
        }
        for (size_t j = 0; j < Settings.iterationsPerRound; ++j) {
            T* r = oi.IterateInstance(vs);
            for (size_t k = 0; k < Settings.outputCount; ++k) {
                T iv = Settings.mask & ~(r[k] ^ rndVal);
                if (std::is_same<int8_t, T>::value || std::is_same<uint8_t, T>::value) {
                    t += BrendanCUDA::Binary::Count1s((uint8_t)iv);
                }
                else if (std::is_same<int16_t, T>::value || std::is_same<uint16_t, T>::value) {
                    t += BrendanCUDA::Binary::Count1s((uint16_t)iv);
                }
                else if (std::is_same<int32_t, T>::value || std::is_same<uint32_t, T>::value) {
                    t += BrendanCUDA::Binary::Count1s((uint32_t)iv);
                }
                else if (std::is_same<int64_t, T>::value || std::is_same<uint64_t, T>::value) {
                    t += BrendanCUDA::Binary::Count1s((uint64_t)iv);
                }
                else {
                    throw std::exception();
                }
            }
            delete[] r;
        }
        delete[] vs;
    }
    T mc;
    if (std::is_same<int8_t, T>::value || std::is_same<uint8_t, T>::value) {
        mc = BrendanCUDA::Binary::Count1s((uint8_t)Settings.mask);
    }
    else if (std::is_same<int16_t, T>::value || std::is_same<uint16_t, T>::value) {
        mc = BrendanCUDA::Binary::Count1s((uint16_t)Settings.mask);
    }
    else if (std::is_same<int32_t, T>::value || std::is_same<uint32_t, T>::value) {
        mc = BrendanCUDA::Binary::Count1s((uint32_t)Settings.mask);
    }
    else if (std::is_same<int64_t, T>::value || std::is_same<uint64_t, T>::value) {
        mc = BrendanCUDA::Binary::Count1s((uint64_t)Settings.mask);
    }
    else {
        throw std::exception();
    }
    return (float)t / (float)((uint64_t)mc * Settings.iterationsPerRound * Settings.roundCount * Settings.outputCount);
}

template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint8_t>(void*, evalProliferation_sd<uint8_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int8_t>(void*, evalProliferation_sd<int8_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint16_t>(void*, evalProliferation_sd<uint16_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int16_t>(void*, evalProliferation_sd<int16_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint32_t>(void*, evalProliferation_sd<uint32_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int32_t>(void*, evalProliferation_sd<int32_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint64_t>(void*, evalProliferation_sd<uint64_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int64_t>(void*, evalProliferation_sd<int64_t>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<float>(void*, evalProliferation_sd<float>&);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<double>(void*, evalProliferation_sd<double>&);