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
            Landmark() = default;
            __host__ __device__ Landmark(uint32_t InputIndex, uint32_t OutputIndex, uint32_t Size) {
                inputIndex = InputIndex;
                outputIndex = OutputIndex;
                size = Size;
            }
        };

        __host__ __device__ static __forceinline ArrayV<Landmark> GetLandmarksInDirection(uint32_t InputLength, uint32_t OutputLength, uint32_t RangeLength, uint32_t InputIndex, uint32_t OutputIndex) {
            Landmark* landmarks = 0;
            size_t landmarkSize = 0;
            size_t landmarkCapacity = 0;

            while (RangeLength) {
                uint32_t dInput = InputLength - InputIndex;
                uint32_t dOutput = OutputLength - OutputIndex;
                bool oG = dOutput > dInput;
                uint32_t dMin = oG ? dInput : dOutput;
                if (dMin >= RangeLength) break;

                if (landmarkCapacity <= landmarkSize + 1) {
                    if (landmarkCapacity == 0) landmarkCapacity = 1;
                    else landmarkCapacity <<= 1;
                    Landmark* newArr = new Landmark[landmarkCapacity];
                    memcpy(newArr, landmarks, landmarkSize * sizeof(Landmark));
                    delete[] landmarks;
                    landmarks = newArr;
                }
                landmarks[landmarkSize++] = Landmark(InputIndex, OutputIndex, dMin);

                if (oG) {
                    InputIndex = 0;
                    OutputIndex += dMin;
                }
                else if (dInput == dOutput) {
                    InputIndex = 0;
                    OutputIndex = 0;
                }
                else {
                    InputIndex += dMin;
                    OutputIndex = 0;
                }
                RangeLength -= dMin;
            }
            ArrayV<Landmark> landmarkArray(landmarkSize);
            if (landmarks) memcpy(landmarkArray.Data(), landmarks, sizeof(Landmark) * landmarkSize);
            return landmarkArray;
        }
    }
    template <typename _T, size_t _VectorLength, bool _InputOnHost, bool _OutputOnHost, bool _Wrap = false>
    __host__ static void CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates);
#ifdef __CUDACC__
    template <typename _T, size_t _VectorLength, bool _Wrap = false>
    __device__ static void CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates);
#endif

    template <typename _T, size_t _VectorLength, copyValFunc_t<_T> _CopyValFunc, bool _Wrap = false>
    __host__ __device__ static void CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates);
    template <typename _T, size_t _VectorLength, copyArrFunc_t<_T> _CopyArrFunc, bool _Wrap = false>
    __host__ __device__ static void CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates);
}

template <typename _T, size_t _VectorLength, bool _InputOnHost, bool _OutputOnHost, bool _Wrap>
__host__ void bcuda::CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
    using vector_t = FixedVector<uint32_t, _VectorLength>;
    
    if constexpr (_Wrap) {
        ArrayV<details::Landmark> landmarksArray[_VectorLength];
        for (size_t i = 0; i < _VectorLength; ++i)
            landmarksArray[i] = details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]);
        
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

            CopyBlock<_T, _VectorLength, _InputOnHost, _OutputOnHost, false>(Input, output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);
            
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
                if constexpr (_OutputOnHost) memcpy(output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x);
                else cudaMemcpy(output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyHostToDevice);
            else
                if constexpr (_OutputOnHost) cudaMemcpy(output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyDeviceToHost);
                else cudaMemcpy(output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x, cudaMemcpyDeviceToDevice);
            return;
        }
        size_t elementNum = RangeDimensions[_VectorLength - 1];
        size_t memcpySize = sizeof(_T) * elementNum;
        vector_t i;
        while (true) {
            size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
            size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

            const _T* iptPtr = Input + iptIdx;
            _T* optPtr = output + optIdx;

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
template <typename _T, size_t _VectorLength, bool _Wrap>
__device__ void bcuda::CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
    using vector_t = FixedVector<uint32_t, _VectorLength>;

    if constexpr (_Wrap) {
        ArrayV<details::Landmark> landmarksArray[_VectorLength];
        for (size_t i = 0; i < _VectorLength; ++i)
            landmarksArray[i] = details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]);
        
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

            CopyBlock<_T, _VectorLength, false>(Input, output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);

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
            memcpy(output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, sizeof(_T) * RangeDimensions.x);
            return;
        }
        size_t elementNum = RangeDimensions[_VectorLength - 1];
        size_t memcpySize = sizeof(_T) * elementNum;
        vector_t i;
        while (true) {
            size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
            size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

            const _T* iptPtr = Input + iptIdx;
            _T* optPtr = output + optIdx;

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

template <typename _T, size_t _VectorLength, bcuda::copyValFunc_t<_T> _CopyValFunc, bool _Wrap>
__host__ __device__ void bcuda::CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
    using vector_t = FixedVector<uint32_t, _VectorLength>;

    if constexpr (_Wrap) {
        ArrayV<details::Landmark> landmarksArray[_VectorLength];
        for (size_t i = 0; i < _VectorLength; ++i)
            landmarksArray[i] = details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]);

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

            CopyBlock<_T, _VectorLength, _CopyValFunc, false>(Input, output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);

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
            for (_T* ipu = Input + RangeDimensions.x; Input < ipu; ++Input, ++output)
                _CopyValFunc(output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x);
            return;
        }
        size_t elementNum = RangeDimensions[_VectorLength - 1];
        vector_t i;
        while (true) {
            size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
            size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

            const _T* iptPtr = Input + iptIdx;
            _T* optPtr = output + optIdx;

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

template <typename _T, size_t _VectorLength, bcuda::copyArrFunc_t<_T> _CopyArrFunc, bool _Wrap>
__host__ __device__ void bcuda::CopyBlock(const _T* Input, _T* output, const FixedVector<uint32_t, _VectorLength>& InputDimensions, const FixedVector<uint32_t, _VectorLength>& OutputDimensions, const FixedVector<uint32_t, _VectorLength>& RangeDimensions, const FixedVector<uint32_t, _VectorLength>& RangeInInputsCoordinates, const FixedVector<uint32_t, _VectorLength>& RangeInOutputsCoordinates) {
    using vector_t = FixedVector<uint32_t, _VectorLength>;

    if constexpr (_Wrap) {
        ArrayV<details::Landmark> landmarksArray[_VectorLength];
        for (size_t i = 0; i < _VectorLength; ++i)
            landmarksArray[i] = details::GetLandmarksInDirection(InputDimensions[i], OutputDimensions[i], RangeDimensions[i], RangeInInputsCoordinates[i], RangeInOutputsCoordinates[i]);

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

            CopyBlock<_T, _VectorLength, _CopyArrFunc, false>(Input, output, InputDimensions, OutputDimensions, rangeDimensions, rangeInInputCoordinates, rangeInOutputCoordinates);

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
            _CopyArrFunc(output + RangeInOutputsCoordinates.x, Input + RangeInInputsCoordinates.x, RangeDimensions.x);
            return;
        }
        size_t elementNum = RangeDimensions[_VectorLength - 1];
        vector_t i;
        while (true) {
            size_t iptIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(InputDimensions, RangeInInputsCoordinates + i);
            size_t optIdx = CoordinatesToIndex<size_t, uint32_t, _VectorLength, true>(OutputDimensions, RangeInOutputsCoordinates + i);

            _CopyArrFunc(output + optIdx, Input + iptIdx, elementNum);

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