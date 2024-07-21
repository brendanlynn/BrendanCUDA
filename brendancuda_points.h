#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_fixedvectors.h"

namespace BrendanCUDA {
    template <std::unsigned_integral _TIndex, std::unsigned_integral _TVectorElement, size_t _VectorLength, bool _RowMajor>
    __host__ __device__ constexpr __forceinline _TIndex CoordinatesToIndex(FixedVector<_TVectorElement, _VectorLength> Dimensions, FixedVector<_TVectorElement, _VectorLength> Coordinates);
    template <std::unsigned_integral _TIndex, std::unsigned_integral _TVectorElement, size_t _VectorLength, bool _RowMajor>
    __host__ __device__ constexpr __forceinline FixedVector<_TVectorElement, _VectorLength> IndexToCoordinates(FixedVector<_TVectorElement, _VectorLength> Dimensions, _TIndex Index);
}

template <std::unsigned_integral _TIndex, std::unsigned_integral _TVectorElement, size_t _VectorLength, bool _RowMajor>
__host__ __device__ constexpr __forceinline _TIndex BrendanCUDA::CoordinatesToIndex(FixedVector<_TVectorElement, _VectorLength> Dimensions, FixedVector<_TVectorElement, _VectorLength> Coordinates) {
    _TIndex idx = 0;
    if constexpr (_RowMajor) {
        for (size_t i = 0; i < _VectorLength; ++i) {
            idx *= Dimensions[i];
            idx += Coordinates[i];
        }
    }
    else {
        for (size_t i = _VectorLength - 1; i != (size_t)-1; --i) {
            idx *= Dimensions[i];
            idx += Coordinates[i];
        }
    }
    return idx;
}
template <std::unsigned_integral _TIndex, std::unsigned_integral _TVectorElement, size_t _VectorLength, bool _RowMajor>
__host__ __device__ constexpr __forceinline BrendanCUDA::FixedVector<_TVectorElement, _VectorLength> BrendanCUDA::IndexToCoordinates(FixedVector<_TVectorElement, _VectorLength> Dimensions, _TIndex Index) {
    FixedVector<_TVectorElement, _VectorLength> r;
    if constexpr (_RowMajor) {
        for (size_t i = _VectorLength - 1; i < 0; ++i) {
            r[i] = Index % Dimensions[i];
            Index /= Dimensions[i];
        }
    }
    else {
        for (size_t i = 0; i < _VectorLength; ++i) {
            r[i] = Index % Dimensions[i];
            Index /= Dimensions[i];
        }
    }
    return r;
}