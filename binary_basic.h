#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace brendancuda {
    namespace binary {
        __host__ __device__ uint32_t CountBitsF(uint64_t Value);
        __host__ __device__ uint32_t CountBitsF(uint32_t Value);
        __host__ __device__ uint32_t CountBitsF(uint16_t Value);
        __host__ __device__ uint32_t CountBitsF(uint8_t Value);

        __host__ __device__ uint32_t CountBitsB(uint64_t Value);
        __host__ __device__ uint32_t CountBitsB(uint32_t Value);
        __host__ __device__ uint32_t CountBitsB(uint16_t Value);
        __host__ __device__ uint32_t CountBitsB(uint8_t Value);
    }
}