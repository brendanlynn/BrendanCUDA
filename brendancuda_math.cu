#include "brendancuda_math.cuh"

#include <cmath>

__host__ __device__ float BrendanCUDA::Math::sqrt(float value) {
    return std::sqrt(value);
}
__host__ __device__ double BrendanCUDA::Math::sqrt(double value) {
    return std::sqrt(value);
}
__host__ __device__ int8_t BrendanCUDA::Math::sqrt(int8_t value) {
    return (int8_t)sqrt((int32_t)value);
}
__host__ __device__ uint8_t BrendanCUDA::Math::sqrt(uint8_t value) {
    return (uint8_t)sqrt((uint32_t)value);
}
__host__ __device__ int16_t BrendanCUDA::Math::sqrt(int16_t value) {
    return (int16_t)sqrt((int32_t)value);
}
__host__ __device__ uint16_t BrendanCUDA::Math::sqrt(uint16_t value) {
    return (uint16_t)sqrt((uint32_t)value);
}
__host__ __device__ int32_t BrendanCUDA::Math::sqrt(int32_t value) {
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
__host__ __device__ uint32_t BrendanCUDA::Math::sqrt(uint32_t value) {
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
__host__ __device__ int64_t BrendanCUDA::Math::sqrt(int64_t value) {
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
__host__ __device__ uint64_t BrendanCUDA::Math::sqrt(uint64_t value) {
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