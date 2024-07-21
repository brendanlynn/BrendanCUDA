#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace Math {
        template <typename _T>
        __host__ __device__ __forceinline _T sqrt(_T value);

        template <typename _T>
        __host__ __device__ __forceinline _T clamp(_T value, _T lower, _T upper);
    }
}
template <>
__host__ __device__ __forceinline float BrendanCUDA::Math::sqrt<float>(float value) {
    return std::sqrt(value);
}
template <>
__host__ __device__ __forceinline double BrendanCUDA::Math::sqrt<double>(double value) {
    return std::sqrt(value);
}
template <>
__host__ __device__ __forceinline int32_t BrendanCUDA::Math::sqrt<int32_t>(int32_t value) {
    if (value < 0) {
        return -1;
    }
    else if (value < 2) {
        return value;
    }

    int32_t lower = 2;
    int32_t upper = value >> 1;

    do {
        int32_t mid = lower + ((upper - lower) >> 1);
        if (value >= mid * mid) {
            lower = mid;
        }
        else {
            upper = mid;
        }
    } while (upper > (lower + 1));

    return lower;
}
template <>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Math::sqrt<uint32_t>(uint32_t value) {
    if (value < 2) {
        return value;
    }

    uint32_t lower = 2;
    uint32_t upper = value >> 1;

    do {
        uint32_t mid = lower + ((upper - lower) >> 1);
        if (value >= mid * mid) {
            lower = mid;
        }
        else {
            upper = mid;
        }
    } while (upper > (lower + 1));

    return lower;
}
template <>
__host__ __device__ __forceinline int8_t BrendanCUDA::Math::sqrt<int8_t>(int8_t value) {
    return (int8_t)sqrt((int32_t)value);
}
template <>
__host__ __device__ __forceinline uint8_t BrendanCUDA::Math::sqrt<uint8_t>(uint8_t value) {
    return (uint8_t)sqrt((uint32_t)value);
}
template <>
__host__ __device__ __forceinline int16_t BrendanCUDA::Math::sqrt<int16_t>(int16_t value) {
    return (int16_t)sqrt((int32_t)value);
}
template <>
__host__ __device__ __forceinline uint16_t BrendanCUDA::Math::sqrt<uint16_t>(uint16_t value) {
    return (uint16_t)sqrt((uint32_t)value);
}
template <>
__host__ __device__ __forceinline int64_t BrendanCUDA::Math::sqrt<int64_t>(int64_t value) {
    if (value < 0) {
        return -1;
    }
    else if (value < 2) {
        return value;
    }

    int64_t lower = 2;
    int64_t upper = value >> 1;

    do {
        int64_t mid = lower + ((upper - lower) >> 1);
        if (value >= mid * mid) {
            lower = mid;
        }
        else {
            upper = mid;
        }
    } while (upper > (lower + 1));

    return lower;
}
template <>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Math::sqrt<uint64_t>(uint64_t value) {
    if (value < 2) {
        return value;
    }

    uint64_t lower = 2;
    uint64_t upper = value >> 1;

    do {
        uint64_t mid = lower + ((upper - lower) >> 1);
        if (value >= mid * mid) {
            lower = mid;
        }
        else {
            upper = mid;
        }
    } while (upper > (lower + 1));

    return lower;
}

template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Math::clamp(_T value, _T lower, _T upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}