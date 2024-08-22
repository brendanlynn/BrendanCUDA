#pragma once

#include "brendancuda_fixedvectors.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_reference.h>

namespace BrendanCUDA {
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

        template <size_t _Index>
            requires (_Index < _DimensionCount)
        __host__ __device__ __forceinline uint32_t Length() const;
        __host__ __device__ __forceinline vector_t Dimensions() const;
    protected:
        __host__ __device__ __forceinline dim3 DimensionsD() const requires (_DimensionCount <= 3);
    public:
        __host__ __device__ __forceinline size_t ValueCount() const;

        __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const;
        __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coordinates) const;
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _DimensionCount)
        __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coordinates) const;
    private:
        vector_t dims;
    };
}

template <size_t _DimensionCount>
BrendanCUDA::DimensionedBase<_DimensionCount>::DimensionedBase(vector_t Dimensions)
    : dims(Dimensions) { }
template <size_t _DimensionCount>
template <std::convertible_to<uint32_t>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
BrendanCUDA::DimensionedBase<_DimensionCount>::DimensionedBase(_Ts... Dimensions)
    : dims(Dimensions...) { }
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::DimensionedBase<_DimensionCount>::LengthX() const requires (_DimensionCount <= 4) {
    return dimensions.x;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::DimensionedBase<_DimensionCount>::LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
    return dimensions.y;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::DimensionedBase<_DimensionCount>::LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
    return dimensions.z;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::DimensionedBase<_DimensionCount>::LengthW() const requires (_DimensionCount == 4) {
    return dimensions.w;
}
template <size_t _DimensionCount>
template <size_t _Index>
    requires (_Index < _DimensionCount)
__host__ __device__ __forceinline uint32_t BrendanCUDA::DimensionedBase<_DimensionCount>::Length() const {
    return dims[_Index];
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::DimensionedBase<_DimensionCount>::Dimensions() const -> vector_t {
    return dims;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline dim3 BrendanCUDA::DimensionedBase<_DimensionCount>::DimensionsD() const requires (_DimensionCount <= 3) {
    if constexpr (_DimensionCount == 1) return dim3(dims.x);
    else if constexpr (_DimensionCount == 2) return dim3(dims.x, dims.y);
    else return dim3(dims.x, dims.y, dims.z);
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline size_t BrendanCUDA::DimensionedBase<_DimensionCount>::ValueCount() const {
    size_t s = 1;
    for (size_t i = 0; i < _DimensionCount; ++i)
        s *= dims[i];
    return s;
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline uint64_t BrendanCUDA::DimensionedBase<_DimensionCount>::CoordsToIdx(vector_t Coords) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Coords);
}
template <size_t _DimensionCount>
template <std::convertible_to<uint32_t>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline uint64_t BrendanCUDA::DimensionedBase<_DimensionCount>::CoordsToIdx(_Ts... Coords) const {
    return CoordsToIdx(vector_t(Coords...));
}
template <size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::DimensionedBase<_DimensionCount>::IdxToCoords(uint64_t Idx) const -> vector_t {
    return BrendanCUDA::IndexToCoordinates<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Idx);
}