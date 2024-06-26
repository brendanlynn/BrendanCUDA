#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace Math {
        template <typename _T>
        __host__ __device__ _T sqrt(_T value);

        template <typename _T>
        __host__ __device__ constexpr _T clamp(_T value, _T lower, _T upper);
    }
}