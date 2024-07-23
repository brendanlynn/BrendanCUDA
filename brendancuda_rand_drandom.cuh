#pragma once

#include "cstdint"
#include "device_launch_parameters.h"
#include "limits"

namespace BrendanCUDA {
    namespace Random {
        class DeviceRandom final {
        public:
#ifdef __CUDACC__
            __device__ DeviceRandom(uint64_t Seed);
#endif

#ifdef __CUDACC__
            __device__ void Iterate();
#endif

#ifdef __CUDACC__
            __device__ float GetF();
            __device__ double GetD();
#endif

#ifdef __CUDACC__
            __device__ uint8_t GetI8();
            __device__ uint16_t GetI16();
            __device__ uint32_t GetI32();
            __device__ uint64_t GetI64();
#endif

#ifdef __CUDACC__
            __device__ uint64_t operator()();
#endif
        private:
            uint64_t c[8];
        };
    }
}