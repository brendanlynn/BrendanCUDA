#include "brendancuda_ai_evolution_evaluation_output_implementations_proliferation.h"
#include "brendancuda_binary_basic.cuh"
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <cmath>

/*template <typename T>
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation(void* Object, void* EvaluationSharedData) {
    constexpr float rndSclrF = 2.f / (float)std::numeric_limits<uint64_t>::max();
    constexpr double rndSclrD = 2. / (double)std::numeric_limits<uint64_t>::max();

    evalProliferation_sd<T> sd = *(evalProliferation_sd<T>*)EvaluationSharedData;
    if (!sd.inputCount) {
        throw std::runtime_error("'sd.inputCount' cannot be zero.");
    }
    instance_v_t<T> oi(sd.instanceFunctions, Object, sd.sd_ci);
    if (std::is_same<T, float>::value) {
        float t = 0.f;
        for (size_t i = 0; i < sd.roundCount; ++i) {
            float* vs = new float[sd.inputCount];
            float rndVal = sd.rng.Run() * rndSclrF - 1.f;
            for (size_t j = 0; j < sd.inputCount; ++j) {
                vs[j] = rndVal;
            }
            for (size_t j = 0; j < sd.iterationsPerRound; ++j) {
                float* r = (float*)oi.IterateInstance((T*)vs);
                float dev_t = 0.f;
                for (size_t k = 0; k < sd.outputCount; ++k) {
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
    else if (std::is_same<T, double>::value) {
        double t = 0.;
        for (size_t i = 0; i < sd.roundCount; ++i) {
            double* vs = new double[sd.inputCount];
            double rndVal = sd.rng.Run() * rndSclrD - 1.;
            for (size_t j = 0; j < sd.inputCount; ++j) {
                vs[j] = rndVal;
            }
            for (size_t j = 0; j < sd.iterationsPerRound; ++j) {
                double* r = (double*)oi.IterateInstance((T*)vs);
                double dev_t = 0.;
                for (size_t k = 0; k < sd.outputCount; ++k) {
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
    else {
        uint64_t t = 0ui64;
        for (size_t i = 0; i < sd.roundCount; ++i) {
            T* vs = new T[sd.inputCount];
            T rndVal = (T)sd.rng.Run();
            for (size_t j = 0; j < sd.inputCount; ++j) {
                vs[j] = rndVal;
            }
            for (size_t j = 0; j < sd.iterationsPerRound; ++j) {
                T* r = oi.IterateInstance(vs);
                T dev_t = 0;
                for (size_t k = 0; k < sd.outputCount; ++k) {
                    T iv = BrendanCUDA::Binary::Count1s(!(r[k] ^ rndVal));
                    dev_t += iv;
                }
                delete[] r;
                t += dev_t;
            }
            delete[] vs;
        }
        return (float)t;
    }
}*/

template <>
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<float>(void* Object, void* EvaluationSharedData) {
    constexpr float rndSclr = 2.f / (float)std::numeric_limits<uint64_t>::max();

    evalProliferation_sd<float> sd = *(evalProliferation_sd<float>*)EvaluationSharedData;
    if (!sd.inputCount) {
        throw std::runtime_error("'sd.inputCount' cannot be zero.");
    }
    instance_v_t<float> oi(sd.instanceFunctions, Object, sd.sd_ci);
    float t = 0.f;
    for (size_t i = 0; i < sd.roundCount; ++i) {
        float* vs = new float[sd.inputCount];
        float rndVal = sd.rng.Run() * rndSclr - 1.f;
        for (size_t j = 0; j < sd.inputCount; ++j) {
            vs[j] = rndVal;
        }
        for (size_t j = 0; j < sd.iterationsPerRound; ++j) {
            float* r = oi.IterateInstance(vs);
            float dev_t = 0.f;
            for (size_t k = 0; k < sd.outputCount; ++k) {
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
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<double>(void* Object, void* EvaluationSharedData) {
    constexpr double rndSclr = 2. / (double)std::numeric_limits<uint64_t>::max();

    evalProliferation_sd<double> sd = *(evalProliferation_sd<double>*)EvaluationSharedData;
    if (!sd.inputCount) {
        throw std::runtime_error("'sd.inputCount' cannot be zero.");
    }
    instance_v_t<double> oi(sd.instanceFunctions, Object, sd.sd_ci);
    double t = 0.;
    for (size_t i = 0; i < sd.roundCount; ++i) {
        double* vs = new double[sd.inputCount];
        double rndVal = sd.rng.Run() * rndSclr - 1.;
        for (size_t j = 0; j < sd.inputCount; ++j) {
            vs[j] = rndVal;
        }
        for (size_t j = 0; j < sd.iterationsPerRound; ++j) {
            double* r = oi.IterateInstance(vs);
            double dev_t = 0.;
            for (size_t k = 0; k < sd.outputCount; ++k) {
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
float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation(void* Object, void* EvaluationSharedData) {
    evalProliferation_sd<T> sd = *(evalProliferation_sd<T>*)EvaluationSharedData;
    if (!sd.inputCount) {
        throw std::runtime_error("'sd.inputCount' cannot be zero.");
    }
    instance_v_t<T> oi(sd.instanceFunctions, Object, sd.sd_ci);
    uint64_t t = 0ui64;
    for (size_t i = 0; i < sd.roundCount; ++i) {
        T* vs = new T[sd.inputCount];
        T rndVal = (T)sd.rng.Run();
        for (size_t j = 0; j < sd.inputCount; ++j) {
            vs[j] = rndVal;
        }
        for (size_t j = 0; j < sd.iterationsPerRound; ++j) {
            T* r = oi.IterateInstance(vs);
            T dev_t = 0;
            for (size_t k = 0; k < sd.outputCount; ++k) {
                T iv = !(r[k] ^ rndVal);
                if (std::is_same<int8_t, T>::value || std::is_same<uint8_t, T>::value) {
                    iv = BrendanCUDA::Binary::Count1s((uint8_t)iv);
                }
                else if (std::is_same<int16_t, T>::value || std::is_same<uint16_t, T>::value) {
                    iv = BrendanCUDA::Binary::Count1s((uint16_t)iv);
                }
                else if (std::is_same<int32_t, T>::value || std::is_same<uint32_t, T>::value) {
                    iv = BrendanCUDA::Binary::Count1s((uint32_t)iv);
                }
                else if (std::is_same<int64_t, T>::value || std::is_same<uint64_t, T>::value) {
                    iv = BrendanCUDA::Binary::Count1s((uint64_t)iv);
                }
                else {
                    throw std::exception();
                }
                dev_t += iv;
            }
            delete[] r;
            t += dev_t;
        }
        delete[] vs;
    }
    return (float)t;
}

template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint8_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int8_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint16_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int16_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint32_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int32_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint64_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int64_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint8_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int8_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint16_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int16_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint32_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int32_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<uint64_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<int64_t>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<float>(void*, void*);
template float BrendanCUDA::AI::Evolution::Evaluation::Output::Implementations::evalProliferation<double>(void*, void*);