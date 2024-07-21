#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace BrendanCUDA {
    namespace Binary {
        __host__ __device__ uint32_t CountBitsF(uint64_t Value);
        __host__ __device__ uint32_t CountBitsF(uint32_t Value);
        __host__ __device__ uint32_t CountBitsF(uint16_t Value);
        __host__ __device__ uint32_t CountBitsF(uint8_t Value);

        __host__ __device__ uint32_t CountBitsB(uint64_t Value);
        __host__ __device__ uint32_t CountBitsB(uint32_t Value);
        __host__ __device__ uint32_t CountBitsB(uint16_t Value);
        __host__ __device__ uint32_t CountBitsB(uint8_t Value);

        [[deprecated("Use std::popcount(...) in header <bit> instead.")]]
        __host__ __device__ uint32_t Count1s(uint64_t Value);
        [[deprecated("Use std::popcount(...) in header <bit> instead.")]]
        __host__ __device__ uint32_t Count1s(uint32_t Value);
        [[deprecated("Use std::popcount(...) in header <bit> instead.")]]
        __host__ __device__ uint32_t Count1s(uint16_t Value);
        [[deprecated("Use std::popcount(...) in header <bit> instead.")]]
        __host__ __device__ uint32_t Count1s(uint8_t Value);
    }
}