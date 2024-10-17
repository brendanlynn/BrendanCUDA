#pragma once

#include "fixedvectors.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace bcuda {
    template <std::unsigned_integral _TIndex, std::unsigned_integral _TVectorElement, size_t _VectorLength, bool _RowMajor>
    __host__ __device__ constexpr inline _TIndex CoordinatesToIndex(FixedVector<_TVectorElement, _VectorLength> Dimensions, FixedVector<_TVectorElement, _VectorLength> Coordinates) {
        _TIndex idx = 0;
        if constexpr (_RowMajor) {
            for (size_t i = _VectorLength - 1; i != (size_t)-1; --i) {
                idx *= Dimensions[i];
                idx += Coordinates[i];
            }
        }
        else {
            for (size_t i = 0; i < _VectorLength; ++i) {
                idx *= Dimensions[i];
                idx += Coordinates[i];
            }
        }
        return idx;
    }
    template <std::unsigned_integral _TIndex, std::unsigned_integral _TVectorElement, size_t _VectorLength, bool _RowMajor>
    __host__ __device__ constexpr inline FixedVector<_TVectorElement, _VectorLength> IndexToCoordinates(FixedVector<_TVectorElement, _VectorLength> Dimensions, _TIndex Index) {
        FixedVector<_TVectorElement, _VectorLength> r;
        if constexpr (_RowMajor) {
            for (size_t i = _VectorLength - 1; i != (size_t)-1; ++i) {
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
}