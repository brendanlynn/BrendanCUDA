#pragma once

#include "cstdint"
#include "device_launch_parameters.h"
#include "limits"

namespace BrendanCUDA {
    class DeviceRandom final {
    public:
        __device__ DeviceRandom(uint64_t Seed);

        __device__ void Iterate();

        __device__ float GetF();
        __device__ double GetD();

        __device__ uint8_t GetI8();
        __device__ uint16_t GetI16();
        __device__ uint32_t GetI32();
        __device__ uint64_t GetI64();

        __device__ uint64_t operator()();
    private:
        uint64_t c[8];
    };
}