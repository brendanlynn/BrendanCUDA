#pragma once

#include "fixedvectors.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_reference.h>

namespace bcuda {
    template <size_t _DimensionCount>
    class DimensionedBase {
        static_assert(_DimensionCount, "_DimensionCount may not be zero.");
    protected:
        using vector_t = FixedVector<uint32_t, _DimensionCount>;
    public:
        __host__ __device__ __forceinline DimensionedBase(vector_t Dimensions) {
            for (size_t i = 0; i < _DimensionCount; ++i)
                if (!Dimensions[i]) {
                    dims = vector_t();
                    return;
                }
            dims = Dimensions;
        }
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _DimensionCount)
        __host__ __device__ __forceinline DimensionedBase(_Ts... Dimensions)
            : DimensionedBase(vector_t(Dimensions...)) { }

        __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4) {
            return dims.x;
        }
        __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
            return dims.y;
        }
        __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
            return dims.z;
        }
        __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4) {
            return dims.w;
        }

        __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
            return dims[Idx];
        }
        __host__ __device__ __forceinline vector_t Dimensions() const {
            return dims;
        }
        __host__ __device__ __forceinline dim3 DimensionsD() const requires (_DimensionCount <= 3) {
            if constexpr (_DimensionCount == 1) return dim3(dims.x);
            else if constexpr (_DimensionCount == 2) return dim3(dims.x, dims.y);
            else return dim3(dims.x, dims.y, dims.z);
        }
        __host__ __device__ __forceinline size_t ValueCount() const {
            size_t s = 1;
            for (size_t i = 0; i < _DimensionCount; ++i)
                s *= dims[i];
            return s;
        }

        __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const {
            return bcuda::IndexToCoordinates<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Idx);
        }
        __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const {
            return bcuda::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Coords);
        }
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _DimensionCount)
        __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
            return CoordsToIdx(vector_t(Coords...));
        }
    private:
        vector_t dims;
    };
}