#pragma once

#include "brendancuda_fixedvectors.h"
#include "brendancuda_points.h"
#include <cuda_runtime.h>

namespace BrendanCUDA {
    template <typename _T, size_t _VectorLength>
    __host__ __device__ void CopyBlock(_T* Input, _T* Output, FixedVector<uint32_t, _VectorLength> InputDimensions, FixedVector<uint32_t, _VectorLength> OutputDimensions, FixedVector<uint32_t, _VectorLength> RangeDimensions, FixedVector<uint32_t, _VectorLength> RangeInInputsCoordinates, FixedVector<uint32_t, _VectorLength> RangeInOutputsCoordinates);
}

template <typename _T, size_t _VectorLength>
__host__ __device__ void BrendanCUDA::CopyBlock(_T* Input, _T* Output, FixedVector<uint32_t, _VectorLength> InputDimensions, FixedVector<uint32_t, _VectorLength> OutputDimensions, FixedVector<uint32_t, _VectorLength> RangeDimensions, FixedVector<uint32_t, _VectorLength> RangeInInputsCoordinates, FixedVector<uint32_t, _VectorLength> RangeInOutputsCoordinates) {
    if constexpr (_VectorLength == 1) {
        cudaMemcpy(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyDefault);
        return;
    }
    size_t memcpySize = sizeof(_T) * RangeDimensions[_VectorLength - 1];
    FixedVector<uint32_t, _VectorLength> i;
    WhileStart:
    while (true) {
        size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
        size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

        cudaMemcpy(Output + optIdx, Input + iptIdx, memcpySize, cudaMemcpyDefault);

        for (size_t j = 0; j < _VectorLength - 1; ++j) {
            uint32_t& si = i[j];
            if (++si >= RangeDimensions[j]) si = 0;
            else goto WhileStart;
        }
        break;
    }
}