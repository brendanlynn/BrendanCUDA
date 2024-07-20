#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_fixedvectors.h"

namespace BrendanCUDA {
    namespace details {
        template <typename _T>
        concept isIntegralCoords32P =
            std::same_as<_T, uint32_1> || std::same_as<_T, uint64_1> ||
            std::same_as<_T, uint32_2> || std::same_as<_T, uint64_2> ||
            std::same_as<_T, uint32_3> || std::same_as<_T, uint64_3>;
        template <typename _T>
        concept isIntegral32P = std::same_as<_T, uint32_t> || std::same_as<_T, uint64_t>;
    }

    template <details::isIntegral32P _TIndex, details::isIntegralCoords32P _TVector, bool _RowMajor>
    __host__ __device__ constexpr __forceinline _TIndex CoordinatesToIndex(_TVector Dimensions, _TVector Coordinates);
    template <details::isIntegral32P _TIndex, details::isIntegralCoords32P _TVector, bool _RowMajor>
    __host__ __device__ constexpr __forceinline _TVector IndexToCoordinates(_TVector Dimensions, _TIndex Index);
}

template <BrendanCUDA::details::isIntegral32P _TIndex, BrendanCUDA::details::isIntegralCoords32P _TVector, bool _RowMajor>
__host__ __device__ constexpr __forceinline _TIndex BrendanCUDA::CoordinatesToIndex(_TVector Dimensions, _TVector Coordinates) {
    if constexpr (std::same_as<_TVector, uint32_1> || std::same_as<_TVector, uint64_1>) {
        return (_TIndex)Coordinates.x;
    }
    else if constexpr (std::same_as<_TVector, uint32_2> || std::same_as<_TVector, uint64_2>) {
        if constexpr (_RowMajor) return (_TIndex)Coordinates.y + (_TIndex)Dimensions.y * (_TIndex)Coordinates.x;
        else return (_TIndex)Coordinates.x + (_TIndex)Dimensions.x * (_TIndex)Coordinates.y;
    }
    else if constexpr (std::same_as<_TVector, uint32_3> || std::same_as<_TVector, uint64_3>) {
        if constexpr (_RowMajor) return (_TIndex)Coordinates.z + (_TIndex)Dimensions.z * ((_TIndex)Coordinates.y + (_TIndex)Dimensions.y * (_TIndex)Coordinates.x);
        else return (_TIndex)Coordinates.x + (_TIndex)Dimensions.x * ((_TIndex)Coordinates.y + (_TIndex)Dimensions.y * (_TIndex)Coordinates.z);
    }
}
template <BrendanCUDA::details::isIntegral32P _TIndex, BrendanCUDA::details::isIntegralCoords32P _TVector, bool _RowMajor>
__host__ __device__ constexpr __forceinline _TVector BrendanCUDA::IndexToCoordinates(_TVector Dimensions, _TIndex Index) {
    if constexpr (std::same_as<_TVector, uint32_1> || std::same_as<_TVector, uint64_1>) {
        return _TVector((uint32_t)Index);
    }
    else if constexpr (std::same_as<_TVector, uint32_2> || std::same_as<_TVector, uint64_2>) {
        _TVector r;
        if constexpr (_RowMajor) {
            r.y = Index % Dimensions.y;
            Index /= Dimensions.y;
            r.x = Index;
        }
        else {
            r.x = Index % Dimensions.x;
            Index /= Dimensions.x;
            r.y = Index;
        }
        return r;
    }
    else if constexpr (std::same_as<_TVector, uint32_3> || std::same_as<_TVector, uint64_3>) {
        _TVector r;
        if constexpr (_RowMajor) {
            r.z = Index % Dimensions.z;
            Index /= Dimensions.z;
            r.y = Index % Dimensions.y;
            Index /= Dimensions.y;
            r.x = Index;
        }
        else {
            r.x = Index % Dimensions.x;
            Index /= Dimensions.x;
            r.y = Index % Dimensions.y;
            Index /= Dimensions.y;
            r.z = Index;
        }
        return r;
    }
}