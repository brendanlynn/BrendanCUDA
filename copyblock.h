#pragma once

#include "arrays.h"
#include "copytype.h"
#include "fixedvectors.h"
#include "points.h"
#include <cuda_runtime.h>

namespace bcuda {
    namespace details {
        struct Landmark {
            uint32_t inputIndex;
            uint32_t outputIndex;
            uint32_t size;
            __host__ __device__ inline constexpr Landmark()
                : inputIndex(0), outputIndex(0), size(0) { }
            __host__ __device__ inline constexpr Landmark(uint32_t InputIndex, uint32_t OutputIndex, uint32_t Size)
                : inputIndex(InputIndex), outputIndex(OutputIndex), size(Size) { }
        };

        __host__ __device__ ArrayV<Landmark> GetLandmarksInDirection(uint32_t InputLength, uint32_t OutputLength, uint32_t RangeLength, uint32_t InputIndex, uint32_t OutputIndex);
    }
    template <typename _T, size_t _VectorLength, bool _InputOnHost, bool _OutputOnHost, bool _Wrap = false>
    __host__ static void CopyBlock(const _T* Input, _T* Output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
        using vector_t = FixedVector<uint32_t, _VectorLength>;

        if constexpr (_Wrap) {
            ArrayV<details::Landmark> landmarksArray[_VectorLength];
            for (size_t i = 0; i < _VectorLength; ++i)
                new (landmarksArray + i) ArrayV<details::Landmark>(details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]));

            vector_t i;
            while (true) {
                vector_t rangeDimensions;
                vector_t rangeInInputCoordinates;
                vector_t rangeInOutputCoordinates;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    details::Landmark landmark = landmarksArray[j][i[j]];
                    rangeDimensions[j] = landmark.size;
                    rangeInInputCoordinates[j] = landmark.inputIndex;
                    rangeInOutputCoordinates[j] = landmark.outputIndex;
                }

                CopyBlock<_T, _VectorLength, _InputOnHost, _OutputOnHost, false>(Input, Output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);

                bool toBreak = true;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    uint32_t& si = i[j];
                    if (++si >= landmarksArray[j].Size()) si = 0;
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
                if constexpr (_InputOnHost)
                    if constexpr (_OutputOnHost) memcpy(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x);
                    else cudaMemcpy(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyHostToDevice);
                else
                    if constexpr (_OutputOnHost) cudaMemcpy(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyDeviceToHost);
                    else cudaMemcpy(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyDeviceToDevice);
                return;
            }
            size_t elementNum = RangeDimensions[0];
            size_t memcpySize = sizeof(_T) * elementNum;
            vector_t i;
            while (true) {
                size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
                size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

                const _T* iptPtr = Input + iptIdx;
                _T* optPtr = Output + optIdx;

                if constexpr (_InputOnHost)
                    if constexpr (_OutputOnHost) memcpy(optPtr, iptPtr, memcpySize);
                    else cudaMemcpy(optPtr, iptPtr, memcpySize, cudaMemcpyHostToDevice);
                else
                    if constexpr (_OutputOnHost) cudaMemcpy(optPtr, iptPtr, memcpySize, cudaMemcpyDeviceToHost);
                    else cudaMemcpy(optPtr, iptPtr, memcpySize, cudaMemcpyDeviceToDevice);

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
#ifdef __CUDACC__
    template <typename _T, size_t _VectorLength, bool _Wrap = false>
    __device__ static void CopyBlock(const _T* Input, _T* Output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
        using vector_t = FixedVector<uint32_t, _VectorLength>;

        if constexpr (_Wrap) {
            ArrayV<details::Landmark> landmarksArray[_VectorLength];
            for (size_t i = 0; i < _VectorLength; ++i)
                new (landmarksArray + i) ArrayV<details::Landmark>(details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]));

            vector_t i;
            while (true) {
                vector_t rangeDimensions;
                vector_t rangeInInputCoordinates;
                vector_t rangeInOutputCoordinates;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    details::Landmark landmark = landmarksArray[j][i[j]];
                    rangeDimensions[j] = landmark.size;
                    rangeInInputCoordinates[j] = landmark.inputIndex;
                    rangeInOutputCoordinates[j] = landmark.outputIndex;
                }

                CopyBlock<_T, _VectorLength, false>(Input, Output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);

                bool toBreak = true;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    uint32_t& si = i[j];
                    if (++si >= landmarksArray[j].Size()) si = 0;
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
                memcpy(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x);
                return;
            }
            size_t elementNum = RangeDimensions[0];
            size_t memcpySize = sizeof(_T) * elementNum;
            vector_t i;
            while (true) {
                size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
                size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

                const _T* iptPtr = Input + iptIdx;
                _T* optPtr = Output + optIdx;

                memcpy(optPtr, iptPtr, memcpySize);

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
#endif

    template <typename _T, size_t _VectorLength, copyValFunc_t<_T> _CopyValFunc, bool _Wrap = false>
    __host__ __device__ static void CopyBlock(const _T* Input, _T* Output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
        using vector_t = FixedVector<uint32_t, _VectorLength>;

        if constexpr (_Wrap) {
            ArrayV<details::Landmark> landmarksArray[_VectorLength];
            for (size_t i = 0; i < _VectorLength; ++i)
                new (landmarksArray + i) ArrayV<details::Landmark>(details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]));

            vector_t i;
            while (true) {
                vector_t rangeDimensions;
                vector_t rangeInInputCoordinates;
                vector_t rangeInOutputCoordinates;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    details::Landmark landmark = landmarksArray[j][i[j]];
                    rangeDimensions[j] = landmark.size;
                    rangeInInputCoordinates[j] = landmark.inputIndex;
                    rangeInOutputCoordinates[j] = landmark.outputIndex;
                }

                CopyBlock<_T, _VectorLength, _CopyValFunc, false>(Input, Output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);

                bool toBreak = true;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    uint32_t& si = i[j];
                    if (++si >= landmarksArray[j].Size()) si = 0;
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
                for (_T* ipu = Input + RangeDimensions.x; Input < ipu; ++Input, ++Output)
                    _CopyValFunc(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x);
                return;
            }
            size_t elementNum = RangeDimensions[0];
            vector_t i;
            while (true) {
                size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
                size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

                const _T* iptPtr = Input + iptIdx;
                _T* optPtr = Output + optIdx;

                for (_T* ipu = iptPtr + elementNum; iptPtr < ipu; ++iptPtr, ++optPtr)
                    _CopyValFunc(optPtr, iptPtr);

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
    template <typename _T, size_t _VectorLength, copyArrFunc_t<_T> _CopyArrFunc, bool _Wrap = false>
    __host__ __device__ static void CopyBlock(const _T* Input, _T* Output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
        using vector_t = FixedVector<uint32_t, _VectorLength>;

        if constexpr (_Wrap) {
            ArrayV<details::Landmark> landmarksArray[_VectorLength];
            for (size_t i = 0; i < _VectorLength; ++i)
                new (landmarksArray + i) ArrayV<details::Landmark>(details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]));

            vector_t i;
            while (true) {
                vector_t rangeDimensions;
                vector_t rangeInInputCoordinates;
                vector_t rangeInOutputCoordinates;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    details::Landmark landmark = landmarksArray[j][i[j]];
                    rangeDimensions[j] = landmark.size;
                    rangeInInputCoordinates[j] = landmark.inputIndex;
                    rangeInOutputCoordinates[j] = landmark.outputIndex;
                }

                CopyBlock<_T, _VectorLength, _CopyArrFunc, false>(Input, Output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);

                bool toBreak = true;
                for (size_t j = 0; j < _VectorLength; ++j) {
                    uint32_t& si = i[j];
                    if (++si >= landmarksArray[j].Size()) si = 0;
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
                _CopyArrFunc(Output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, RangeDimensions.x);
                return;
            }
            size_t elementNum = RangeDimensions[0];
            vector_t i;
            while (true) {
                size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
                size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

                _CopyArrFunc(Output + optIdx, Input + iptIdx, elementNum);

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
}