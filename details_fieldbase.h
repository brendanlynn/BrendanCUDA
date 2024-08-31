#pragma once

#include "copyblock.h"
#include "copytype.h"
#include "cudaconstexpr.h"
#include "details_fillwith.h"
#include "dimensionedbase.h"
#include "errorhelp.h"
#include "points.h"
#include "BSerializer/Serializer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/device_ptr.h>
#include <type_traits>

namespace BrendanCUDA {
    namespace details {
        template <typename _T, size_t _DimensionCount>
        class FieldBase : public DimensionedBase<_DimensionCount> {
            using this_t = FieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        protected:
            using vector_t = basedb_t::vector_t;
        public:
#pragma region Constructors
            __host__ __device__ __forceinline FieldBase(const vector_t& Dimensions);
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline FieldBase(_Ts... Dimensions)
                : FieldBase(vector_t(Dimensions...)) { }
            __host__ __device__ __forceinline FieldBase(const vector_t& Dimensions, _T* Arr)
                : basedb_t(Dimensions), darr(Arr) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline FieldBase(_Ts... Dimensions, _T* Arr)
                : FieldBase(vector_t(Dimensions...), Arr) { }
#pragma endregion

            __host__ __device__ __forceinline size_t SizeOnGPU() const {
                return sizeof(_T) * basedb_t::ValueCount();
            }

#pragma region RefConversion
            __host__ __device__ __forceinline _T* IdxToPtr(uint64_t Idx) const;
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> IdxToRef(uint64_t Idx) const;
            __host__ __device__ __forceinline _T* CoordsToPtr(const vector_t& Coords) const;
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline _T* CoordsToPtr(_Ts... Coords) const {
                return CoordsToPtr(vector_t(Coords...));
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> CoordsToRef(const vector_t& Coords) const;
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> CoordsToRef(_Ts... Coords) const {
                return CoordsToRef(vector_t(Coords...));
            }
            __host__ __device__ __forceinline uint64_t PtrToIdx(const _T* Ptr) const;
            __host__ __device__ __forceinline vector_t PtrToCoords(const _T* Ptr) const;
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> PtrToRef(const _T* Ptr) const;
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> PtrToRef(const _T* Ptr);
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<_T> Ref) const;
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<const _T> Ref) const;
#ifdef __CUDACC__
            __device__ __forceinline uint64_t RefToIdx(const _T& Ref) const;
#endif
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<_T> Ref) const;
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<const _T> Ref) const;
#ifdef __CUDACC__
            __device__ __forceinline vector_t RefToCoords(const _T& Ref) const;
#endif
            __host__ __forceinline _T* RefToPtr(thrust::device_reference<_T> Ref) const;
            __host__ __forceinline _T* RefToPtr(thrust::device_reference<const _T> Ref) const;
#ifdef __CUDACC__
            __device__ __forceinline _T* RefToPtr(const _T& Ref) const;
#endif
#pragma endregion

#pragma region OperatorInvoke
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(uint64_t Idx) const;
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(const vector_t& Coords) const;
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(_Ts... Coords) const {
                return CoordsToRef(Coords...);
            }
#pragma endregion

#pragma region CpyAll
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyAllIn(const _T* All) const;
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllIn(const _T* All) const;
#endif
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const;
#ifdef __CUDACC__
            __device__ __forceinline _T* CpyAllOut() const;
#endif
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const;
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllOut(_T* All) const;
#endif
#pragma endregion

#pragma region CpyVal
            __host__ __device__ __forceinline void CpyValIn(uint64_t Idx, const _T& Val) const;
            __host__ __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T& Val) const;
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) const;
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) const;
#endif
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) const;
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) const;
#endif
            __host__ __device__ __forceinline void CpyValOut(uint64_t Idx, _T& Val) const;
            __host__ __device__ __forceinline void CpyValOut(const vector_t& Coords, _T& Val) const;
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const;
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const;
#endif
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const;
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const;
#endif
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const;
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const;
#pragma endregion

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline _T* Data() const;

            template <bool _InputOnHost>
            __host__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const;
#endif
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const;
#endif
            
            __host__ __device__ this_t Clone() const;

            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T>;
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T>;
            static __forceinline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T>;
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T>;
        private:
            _T* darr;
        };
    }
}

template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline BrendanCUDA::details::FieldBase<_T, _DimensionCount>::FieldBase(const vector_t& Dimensions)
    : basedb_t(Dimensions) {
    if (!(this->Length(0))) {
        darr = 0;
        return;
    }
#ifdef __CUDA_ARCH__
    darr = (_T*)malloc(SizeOnGPU());
#else
    ThrowIfBad(cudaMalloc(&darr, SizeOnGPU()));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::IdxToPtr(uint64_t Idx) const {
    return darr + Idx;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::details::FieldBase<_T, _DimensionCount>::IdxToRef(uint64_t Idx) const {
#ifdef __CUDA_ARCH__
    return darr[Idx];
#else
    return *thrust::device_ptr<_T>(darr + Idx);
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CoordsToPtr(const vector_t& Coords) const {
    return IdxToPtr(this->CoordsToIdx(Coords));
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CoordsToRef(const vector_t& Coords) const {
    return IdxToRef(this->CoordsToIdx(Coords));
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint64_t BrendanCUDA::details::FieldBase<_T, _DimensionCount>::PtrToIdx(const _T* Ptr) const {
    return Ptr - darr;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::details::FieldBase<_T, _DimensionCount>::PtrToCoords(const _T* Ptr) const -> vector_t {
    return this->IdxToCoords(PtrToIdx(Ptr));
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::details::FieldBase<_T, _DimensionCount>::PtrToRef(const _T* Ptr) const {
#ifdef __CUDA_ARCH__
    return *Ptr;
#else
    return *thrust::device_ptr<_T>(Ptr);
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline uint64_t BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToIdx(thrust::device_reference<const _T> Ref) const {
    return PtrToIdx(RefToPtr(Ref));
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline uint64_t BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToIdx(thrust::device_reference<_T> Ref) const {
    return PtrToIdx(RefToPtr(Ref));
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline uint64_t BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToIdx(const _T& Ref) const {
    return PtrToIdx(RefToPtr(Ref));
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __forceinline auto BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToCoords(thrust::device_reference<_T> Ref) const -> vector_t {
    return PtrToCoords(RefToPtr(Ref));
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline auto BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToCoords(thrust::device_reference<const _T> Ref) const -> vector_t {
    return PtrToCoords(RefToPtr(Ref));
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline auto BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToCoords(const _T& Ref) const -> vector_t {
    return PtrToCoords(RefToPtr(Ref));
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToPtr(thrust::device_reference<_T> Ref) const {
    return (&Ref).get();
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToPtr(thrust::device_reference<const _T> Ref) const {
    return (&Ref).get();
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::RefToPtr(const _T& Ref) const {
    return const_cast<_T*>(&Ref);
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::details::FieldBase<_T, _DimensionCount>::operator()(uint64_t Idx) const {
    return IdxToRef(Idx);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::details::FieldBase<_T, _DimensionCount>::operator()(const vector_t& Coords) const {
    return CoordsToRef(Coords);
}
template <typename _T, size_t _DimensionCount>
template <bool _CopyFromHost>
__host__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyAllIn(const _T* All) const {
    cudaMemcpy(darr, All, SizeOnGPU(), _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyAllIn(const _T* All) const {
    memcpy(darr, All, SizeOnGPU());
}
#endif
template <typename _T, size_t _DimensionCount>
template <bool _CopyToHost>
__host__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyAllOut() const {
    _T* all;
    if constexpr (_CopyToHost) cudaMalloc(&all, SizeOnGPU());
    else malloc(all);

    CpyAllOut<_CopyToHost>(all);

    return all;
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyAllOut() const {
    _T* all = malloc(SizeOnGPU());

    CpyAllOut(all);

    return all;
}
#endif
template <typename _T, size_t _DimensionCount>
template <bool _CopyToHost>
__host__ __device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyAllOut(_T* All) const {
    cudaMemcpy(All, darr, SizeOnGPU(), _CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyAllOut(_T* All) const {
    memcpy(All, darr, SizeOnGPU());
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValIn(uint64_t Idx, const _T& Val) const {
#ifdef __CUDA_ARCH__
    memcpy(IdxToPtr(Idx), &Val, sizeof(_T));
#else
    cudaMemcpy(IdxToPtr(Idx), &Val, sizeof(_T), cudaMemcpyHostToDevice);
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValIn(const vector_t& Coords, const _T& Val) const {
#ifdef __CUDA_ARCH__
    memcpy(CoordsToPtr(Coords), &Val, sizeof(_T));
#else
    cudaMemcpy(CoordsToPtr(Coords), &Val, sizeof(_T), cudaMemcpyHostToDevice);
#endif
}
template <typename _T, size_t _DimensionCount>
template <bool _CopyFromHost>
__host__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValIn(uint64_t Idx, const _T* Val) const {
    cudaMemcpy(IdxToPtr(Idx), Val, sizeof(_T), _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValIn(uint64_t Idx, const _T* Val) const {
    memcpy(IdxToPtr(Idx), Val, sizeof(_T));
}
#endif
template <typename _T, size_t _DimensionCount>
template <bool _CopyFromHost>
__host__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValIn(const vector_t& Coords, const _T* Val) const {
    cudaMemcpy(CoordsToPtr(Coords), Val, sizeof(_T), _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValIn(const vector_t& Coords, const _T* Val) const {
    memcpy(CoordsToPtr(Coords), &Val, sizeof(_T));
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(uint64_t Idx, _T& Val) const {
#ifdef __CUDA_ARCH__
    memcpy(&Val, IdxToPtr(Idx), sizeof(_T));
#else
    cudaMemcpy(&Val, IdxToPtr(Idx), sizeof(_T), cudaMemcpyDeviceToHost);
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(const vector_t& Coords, _T& Val) const {
#ifdef __CUDA_ARCH__
    memcpy(&Val, CoordsToPtr(Coords), sizeof(_T));
#else
    cudaMemcpy(&Val, CoordsToPtr(Coords), sizeof(_T), cudaMemcpyDeviceToHost);
#endif
}
template <typename _T, size_t _DimensionCount>
template <bool _CopyToHost>
__host__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(uint64_t Idx, _T* Val) const {
    cudaMemcpy(Val, IdxToPtr(Idx), sizeof(_T), _CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(uint64_t Idx, _T* Val) const {
    memcpy(Val, IdxToPtr(Idx), sizeof(_T));
}
#endif
template <typename _T, size_t _DimensionCount>
template <bool _CopyToHost>
__host__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(const vector_t& Coords, _T* Val) const {
    cudaMemcpy(Val, CoordsToPtr(Coords), sizeof(_T), _CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(const vector_t& Coords, _T* Val) const {
    memcpy(Val, CoordsToPtr(Coords), sizeof(_T));
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(uint64_t Idx) const {
    _T val;
    CpyValOut(Idx, val);
    return val;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CpyValOut(const vector_t& Coords) const {
    _T val;
    CpyValOut(Coords, val);
    return val;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::Dispose() {
#ifdef __CUDA_ARCH__
    free(darr);
#else
    ThrowIfBad(cudaFree(darr));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T* BrendanCUDA::details::FieldBase<_T, _DimensionCount>::Data() const {
    return darr;
}
template <typename _T, size_t _DimensionCount>
template <bool _InputOnHost>
__host__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
    CopyBlock<_T, _DimensionCount, _InputOnHost, false, true>(Input, darr, InputDimensions, this->Dimensions(), RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
    CopyBlock<_T, _DimensionCount, true>(Input, darr, InputDimensions, this->Dimensions(), RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
#endif
template <typename _T, size_t _DimensionCount>
template <bool _OutputOnHost>
__host__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
    CopyBlock<_T, _DimensionCount, false, _OutputOnHost, true>(darr, Output, this->Dimensions(), OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
    CopyBlock<_T, _DimensionCount, true>(darr, Output, this->Dimensions(), OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __device__ auto BrendanCUDA::details::FieldBase<_T, _DimensionCount>::Clone() const -> this_t {
    return FieldBase<_T, _DimensionCount>(this->Dimensions(), darr);
}
template <typename _T, size_t _DimensionCount>
__forceinline size_t BrendanCUDA::details::FieldBase<_T, _DimensionCount>::SerializedSize() const requires BSerializer::Serializable<_T> {
    size_t t = sizeof(uint32_t) * _DimensionCount;
    size_t l = this->ValueCount();
    for (size_t i = 0; i < l; ++i)
        t += BSerializer::SerializedSize(CpyValOut(i));
    return t;
}
template <typename _T, size_t _DimensionCount>
__forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
    BSerializer::Serialize<vector_t>(Data, this->Dimensions());
    size_t l = this->ValueCount();
    for (size_t i = 0; i < l; ++i)
        BSerializer::Serialize(Data, CpyValOut(i));
}
template <typename _T, size_t _DimensionCount>
__forceinline auto BrendanCUDA::details::FieldBase<_T, _DimensionCount>::Deserialize(const void*& Data) -> this_t requires BSerializer::Serializable<_T> {
    vector_t dimensions = BSerializer::Deserialize<vector_t>(Data);
    FieldBase<_T, _DimensionCount> field(dimensions);
    size_t l = field.ValueCount();
    for (size_t i = 0; i < l; ++i)
        field.SetValueAt(i, BSerializer::Deserialize<_T>(Data));
    return field;
}
template <typename _T, size_t _DimensionCount>
__forceinline void BrendanCUDA::details::FieldBase<_T, _DimensionCount>::Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
    new (Value) FieldBase<_T, _DimensionCount>(Deserialize(Data));
}