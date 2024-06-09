#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace Math {
        __host__ __device__ float sqrt(float value);
        __host__ __device__ double sqrt(double value);
        __host__ __device__ int8_t sqrt(int8_t value);
        __host__ __device__ uint8_t sqrt(uint8_t value);
        __host__ __device__ int16_t sqrt(int16_t value);
        __host__ __device__ uint16_t sqrt(uint16_t value);
        __host__ __device__ int32_t sqrt(int32_t value);
        __host__ __device__ uint32_t sqrt(uint32_t value);
        __host__ __device__ int64_t sqrt(int64_t value);
        __host__ __device__ uint64_t sqrt(uint64_t value);
    }
}