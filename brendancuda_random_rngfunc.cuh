#pragma once

#include "brendancuda_random_devicerandom.cuh"
#include <random>

namespace BrendanCUDA {
    namespace Random {
        template <typename T>
        using rngFunc_t = T(*)(void*);

        template <typename T>
        class rngWState final {
        public:
            rngFunc_t<T> func;
            void* state;

            __host__ __device__ rngWState(rngFunc_t<T> func, void* state);

            __host__ __device__ T Run() const;
        };

        __device__ rngWState<uint64_t> rngWState64_FromDeviceRandom(DeviceRandom* dr);
        __device__ rngWState<uint32_t> rngWState32_FromDeviceRandom(DeviceRandom* dr);
        rngWState<uint64_t> rngWState64_From_mt19937(std::mt19937* dr);
        rngWState<uint32_t> rngWState32_From_mt19937(std::mt19937* dr);
        rngWState<uint64_t> rngWState64_From_mt19937_64(std::mt19937_64* dr);
        rngWState<uint32_t> rngWState32_From_mt19937_64(std::mt19937_64* dr);

        class rngWStateA final {
            rngFunc_t<float> funcF;
            rngFunc_t<double> funcD;
            rngFunc_t<uint8_t> func8;
            rngFunc_t<uint16_t> func16;
            rngFunc_t<uint32_t> func32;
            rngFunc_t<uint64_t> func64;
            void* state;

            __host__ __device__ rngWStateA(rngFunc_t<float> funcF, rngFunc_t<double> funcD, rngFunc_t<uint8_t> func8, rngFunc_t<uint16_t> func16, rngFunc_t<uint32_t> func32, rngFunc_t<uint64_t> func64, void* state);

            __host__ __device__ float RunF();
            __host__ __device__ double RunD();
            __host__ __device__ uint8_t Run8();
            __host__ __device__ uint16_t Run16();
            __host__ __device__ uint32_t Run32();
            __host__ __device__ uint64_t Run64();
        };
    }
}