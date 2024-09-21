#pragma once

#include <cstdint>

namespace bcuda {
    enum CopyType : uint32_t {
        copyTypeMemcpy,
        copyTypeCopyAssignment,
        copyTypeCopyPlacementNew,
        copyTypeMoveAssignment,
        copyTypeMovePlacementNew
    };

    template <typename _T>
    using copyValFunc_t = void(*)(_T* DestPtr, _T* SourcePtr);
    template <typename _T>
    using copyArrFunc_t = void(*)(_T* DestPtr, _T* SourcePtr, size_t Count);

    template <bool _DestOnHost, bool _SourceOnHost, typename _T>
    __host__ void CopyFunc_Memcpy(_T* DestPtr, _T* SourcePtr) {
        if constexpr (_DestOnHost)
            if constexpr (_SourceOnHost) memcpy(DestPtr, SourcePtr, sizeof(_T));
            else cudaMemcpy(DestPtr, SourcePtr, sizeof(_T), cudaMemcpyDeviceToHost);
        else
            if constexpr (_SourceOnHost) cudaMemcpy(DestPtr, SourcePtr, sizeof(_T), cudaMemcpyHostToDevice);
            else cudaMemcpy(DestPtr, SourcePtr, sizeof(_T), cudaMemcpyDeviceToDevice);
    }
#ifdef __CUDACC__
    template <typename _T>
    __device__ void CopyFunc_Memcpy(_T* DestPtr, _T* SourcePtr) {
        memcpy(DestPtr, SourcePtr, sizeof(_T));
    }
#endif
    template <bool _DestOnHost, bool _SourceOnHost, typename _T>
    __host__ static __forceinline void CopyArrFunc_Memcpy(_T* DestPtr, _T* SourcePtr, size_t Count) {
        if constexpr (_DestOnHost)
            if constexpr (_SourceOnHost) memcpy(DestPtr, SourcePtr, Count * sizeof(_T));
            else cudaMemcpy(DestPtr, SourcePtr, Count * sizeof(_T), cudaMemcpyDeviceToHost);
        else
            if constexpr (_SourceOnHost) cudaMemcpy(DestPtr, SourcePtr, Count * sizeof(_T), cudaMemcpyHostToDevice);
            else cudaMemcpy(DestPtr, SourcePtr, Count * sizeof(_T), cudaMemcpyDeviceToDevice);
    }
#ifdef __CUDACC__
    template <typename _T>
    __device__ void CopyArrFunc_Memcpy(_T* DestPtr, _T* SourcePtr, size_t Count) {
        memcpy(DestPtr, SourcePtr, Count * sizeof(_T));
    }
#endif

    template <typename _T>
    __host__ __device__ void CopyFunc_CopyAssignment(_T* DestPtr, _T* SourcePtr) {
        *DestPtr = *SourcePtr;
    }
    template <typename _T>
    __host__ __device__ void CopyArrFunc_CopyAssignment(_T* DestPtr, _T* SourcePtr, size_t Count) {
        for (_T* spu = SourcePtr + Count; SourcePtr < spu; ++DestPtr, ++SourcePtr)
            *DestPtr = *SourcePtr;
    }

    template <typename _T>
    __host__ __device__ void CopyFunc_CopyPlacementNew(_T* DestPtr, _T* SourcePtr) {
        new (DestPtr) _T(*SourcePtr);
    }
    template <typename _T>
    __host__ __device__ void CopyArrFunc_CopyPlacementNew(_T* DestPtr, _T* SourcePtr, size_t Count) {
        for (_T* spu = SourcePtr + Count; SourcePtr < spu; ++DestPtr, ++SourcePtr)
            new (DestPtr) _T(*SourcePtr);
    }

    template <typename _T>
    __host__ __device__ void CopyFunc_MoveAssignment(_T* DestPtr, _T* SourcePtr) {
        *DestPtr = std::move(*SourcePtr);
    }
    template <typename _T>
    __host__ __device__ void CopyArrFunc_MoveAssignment(_T* DestPtr, _T* SourcePtr, size_t Count) {
        for (_T* spu = SourcePtr + Count; SourcePtr < spu; ++DestPtr, ++SourcePtr)
            *DestPtr = std::move(*SourcePtr);
    }

    template <typename _T>
    __host__ __device__ void CopyFunc_MovePlacementNew(_T* DestPtr, _T* SourcePtr) {
        new (DestPtr) _T(std::move(*SourcePtr));
    }
    template <typename _T>
    __host__ __device__ void CopyArrFunc_MovePlacementNew(_T* DestPtr, _T* SourcePtr, size_t Count) {
        for (_T* spu = SourcePtr + Count; SourcePtr < spu; ++DestPtr, ++SourcePtr)
            new (DestPtr) _T(std::move(*SourcePtr));
    }
}