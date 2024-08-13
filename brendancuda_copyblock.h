#pragma once

#include "brendancuda_arrays.h"
#include "brendancuda_fixedvectors.h"
#include "brendancuda_points.h"
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace details {
        struct Landmark {
            uint32_t inputIndex;
            uint32_t outputIndex;
            uint32_t size;
            Landmark() = default;
            __host__ __device__ Landmark(uint32_t InputIndex, uint32_t OutputIndex, uint32_t Size) {
                inputIndex = InputIndex;
                outputIndex = OutputIndex;
                size = Size;
            }
        };

        __host__ __device__ ArrayV<Landmark> GetLandmarksInDirection(uint32_t InputLength, uint32_t OutputLength, uint32_t RangeLength, uint32_t InputIndex, uint32_t OutputIndex);
    }
    template <typename _T, size_t _VectorLength, bool _Wrap = false>
    __host__ __device__ void CopyBlock(_T* Input, _T* Output, FixedVector<uint32_t, _VectorLength> InputDimensions, FixedVector<uint32_t, _VectorLength> OutputDimensions, FixedVector<uint32_t, _VectorLength> RangeDimensions, FixedVector<uint32_t, _VectorLength> RangeInInputsCoordinates, FixedVector<uint32_t, _VectorLength> RangeInOutputsCoordinates);
}

template <typename _T, size_t _VectorLength, bool _Wrap>
__host__ __device__ void BrendanCUDA::CopyBlock(_T* Input, _T* Output, FixedVector<uint32_t, _VectorLength> InputDimensions, FixedVector<uint32_t, _VectorLength> OutputDimensions, FixedVector<uint32_t, _VectorLength> RangeDimensions, FixedVector<uint32_t, _VectorLength> RangeInInputsCoordinates, FixedVector<uint32_t, _VectorLength> RangeInOutputsCoordinates) {
    if constexpr (_Wrap) {
        ArrayF<ArrayV<details::Landmark>, _VectorLength> landmarksArray;
        for (size_t i = 0; i < _VectorLength; ++i) {
            landmarksArray[i] = details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]);
        }

        FixedVector<uint32_t, _VectorLength> i;
        while (true) {
            FixedVector<uint32_t, _VectorLength> rangeDimensions;
            FixedVector<uint32_t, _VectorLength> rangeInInputCoordinates;
            FixedVector<uint32_t, _VectorLength> rangeInOutputCoordinates;
            for (size_t j = 0; j < _VectorLength; ++j) {
                details::Landmark landmark = landmarksArray[j][i[j]];
                rangeDimensions[j] = landmark.size;
                rangeInInputCoordinates[j] = landmark.inputIndex;
                rangeInOutputCoordinates[j] = landmark.outputIndex;
            }

            CopyBlock<_T, _VectorLength>(Input, Output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);
            
            bool toBreak = true;
            for (size_t j = 0; j < _VectorLength; ++j) {
                uint32_t& si = i[j];
                if (++si >= landmarksArray[j].size) si = 0;
                else {
                    toBreak = false;
                    break;
                }
            }
            if (toBreak) break;
        }
    }
    else {
        if constexpr (_VectorLength == 1) {
            cudaMemcpy(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyDefault);
            return;
        }
        size_t memcpySize = sizeof(_T) * RangeDimensions[_VectorLength - 1];
        FixedVector<uint32_t, _VectorLength> i;
        while (true) {
            size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
            size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

            cudaMemcpy(Output + optIdx, Input + iptIdx, memcpySize, cudaMemcpyDefault);

            bool toBreak = true;
            for (size_t j = 1; j < _VectorLength; ++j) {
                uint32_t& si = i[j];
                if (++si >= RangeDimensions[j]) si = 0;
                else {
                    toBreak = false;
                    break;
                }
            }
            if (toBreak) break;
        }
    }
}