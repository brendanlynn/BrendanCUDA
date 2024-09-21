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
        __host__ __device__ __forceinline DimensionedBase(vector_t Dimensions);
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _DimensionCount)
        __host__ __device__ __forceinline DimensionedBase(_Ts... Dimensions)
            : DimensionedBase(vector_t(Dimensions...)) { }

        __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4);
        __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4);
        __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4);
        __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4);

        __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
            return dims[Idx];
        }
        __host__ __device__ __forceinline vector_t Dimensions() const;
        __host__ __device__ __forceinline dim3 DimensionsD() const requires (_DimensionCount <= 3);
        __host__ __device__ __forceinline size_t ValueCount() const;

        __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const;
        __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const;
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _DimensionCount)
        __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
            return CoordsToIdx(vector_t(Coords...));
        }
    private:
        vector_t dims;
    };
}

template <size_t _DimensionCount>
bcuda::DimensionedBase<_DimensionCount>::DimensionedBase(vector_t Dimensions) {
    for (size_t i = 0; i < _DimensionCount; ++i)
        if (!Dimensions[i]) {
            dims = vector_t();
            return;
        }
    dims = Dimensions;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t bcuda::DimensionedBase<_DimensionCount>::LengthX() const requires (_DimensionCount <= 4) {
    return dims.x;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t bcuda::DimensionedBase<_DimensionCount>::LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
    return dims.y;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t bcuda::DimensionedBase<_DimensionCount>::LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
    return dims.z;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t bcuda::DimensionedBase<_DimensionCount>::LengthW() const requires (_DimensionCount == 4) {
    return dims.w;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline auto bcuda::DimensionedBase<_DimensionCount>::Dimensions() const -> vector_t {
    return dims;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline dim3 bcuda::DimensionedBase<_DimensionCount>::DimensionsD() const requires (_DimensionCount <= 3) {
    if constexpr (_DimensionCount == 1) return dim3(dims.x);
    else if constexpr (_DimensionCount == 2) return dim3(dims.x, dims.y);
    else return dim3(dims.x, dims.y, dims.z);
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline size_t bcuda::DimensionedBase<_DimensionCount>::ValueCount() const {
    size_t s = 1;
    for (size_t i = 0; i < _DimensionCount; ++i)
        s *= dims[i];
    return s;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint64_t bcuda::DimensionedBase<_DimensionCount>::CoordsToIdx(vector_t Coords) const {
    return bcuda::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Coords);
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline auto bcuda::DimensionedBase<_DimensionCount>::IdxToCoords(uint64_t Idx) const -> vector_t {
    return bcuda::IndexToCoordinates<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Idx);
}