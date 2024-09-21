#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <cuda_runtime.h>

namespace brendancuda {
    namespace Math {
        template <typename _T>
        __host__ __device__ static __forceinline _T sqrt(_T value);

        template <typename _T>
        __host__ __device__ static __forceinline _T clamp(_T value, _T lower, _T upper);
    }
}

template <>
__host__ __device__ static __forceinline int32_t brendancuda::Math::sqrt<int32_t>(int32_t value) {
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
__host__ __device__ static __forceinline uint32_t brendancuda::Math::sqrt<uint32_t>(uint32_t value) {
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
__host__ __device__ static __forceinline int64_t brendancuda::Math::sqrt<int64_t>(int64_t value) {
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
__host__ __device__ static __forceinline uint64_t brendancuda::Math::sqrt<uint64_t>(uint64_t value) {
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
__host__ __device__ static __forceinline _T brendancuda::Math::sqrt(_T value) {
    if constexpr (std::signed_integral<_T>)
        return (_T)sqrt((int32_t)value);
    else if constexpr (std::unsigned_integral<_T>)
        return (_T)sqrt((uint32_t)value);
    else
        return std::sqrt(value);
}

template <typename _T>
__host__ __device__ static __forceinline _T brendancuda::Math::clamp(_T value, _T lower, _T upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}