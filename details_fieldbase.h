#pragma once

#include "BSerializer/Serializer.h"
#include "copyblock.h"
#include "copytype.h"
#include "cudaconstexpr.h"
#include "details_fillwith.h"
#include "dimensionedbase.h"
#include "errorhelp.h"
#include "points.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/device_ptr.h>
#include <type_traits>

namespace bcuda {
    namespace details {
        template <typename _T, size_t _DimensionCount>
        class FieldBase : public DimensionedBase<_DimensionCount> {
            using this_t = FieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        public:
#pragma region Constructors
            __host__ __device__ inline FieldBase(const typename this_t::vector_t& Dimensions)
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
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ inline FieldBase(_Ts... Dimensions)
                : FieldBase(typename this_t::vector_t(Dimensions...)) { }
            __host__ __device__ inline FieldBase(const typename this_t::vector_t& Dimensions, _T* Arr)
                : basedb_t(Dimensions), darr(Arr) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ inline FieldBase(_Ts... Dimensions, _T* Arr)
                : FieldBase(typename this_t::vector_t(Dimensions...), Arr) { }
#pragma endregion

            __host__ __device__ inline size_t SizeOnGPU() const {
                return sizeof(_T) * basedb_t::ValueCount();
            }

#pragma region RefConversion
            __host__ __device__ inline _T* IdxToPtr(uint64_t Idx) const {
                return darr + Idx;
            }
            __host__ __device__ inline _T* CoordsToPtr(const typename this_t::vector_t& Coords) const {
                return IdxToPtr(this->CoordsToIdx(Coords));
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ inline _T* CoordsToPtr(_Ts... Coords) const {
                return CoordsToPtr(typename this_t::vector_t(Coords...));
            }
            __host__ __device__ inline uint64_t PtrToIdx(const _T* Ptr) const {
                return Ptr - darr;
            }
            __host__ __device__ inline typename this_t::vector_t PtrToCoords(const _T* Ptr) const {
                return this->IdxToCoords(PtrToIdx(Ptr));
            }
#pragma endregion

#pragma region OperatorInvoke
            __device__ inline _T& operator()(uint64_t Idx) const {
                return *IdxToPtr(Idx);
            }
            __device__ inline _T& operator()(const typename this_t::vector_t& Coords) const {
                return *CoordsToPtr(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __device__ inline _T& operator()(_Ts... Coords) const {
                return *CoordsToPtr(typename this_t::vector_t(Coords...));
            }
#pragma endregion

#pragma region CpyAll
            template <bool _CopyFromHost>
            __host__ inline void CpyAllIn(const _T* All) const {
                ThrowIfBad(cudaMemcpy(darr, All, SizeOnGPU(), _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
            }
#ifdef __CUDACC__
            __device__ inline void CpyAllIn(const _T* All) const {
                memcpy(darr, All, SizeOnGPU());
            }
#endif
            template <bool _CopyToHost>
            __host__ inline _T* CpyAllOut() const {
                _T* all;
                if constexpr (_CopyToHost) ThrowIfBad(cudaMalloc(&all, SizeOnGPU()));
                else malloc(all);

                CpyAllOut<_CopyToHost>(all);

                return all;
            }
#ifdef __CUDACC__
            __device__ inline _T* CpyAllOut() const {
                _T* all = malloc(SizeOnGPU());

                CpyAllOut(all);

                return all;
            }
#endif
            template <bool _CopyToHost>
            __host__ __device__ inline void CpyAllOut(_T* All) const {
                ThrowIfBad(cudaMemcpy(All, darr, SizeOnGPU(), _CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
            }
#ifdef __CUDACC__
            __device__ inline void CpyAllOut(_T* All) const {
                memcpy(All, darr, SizeOnGPU());
            }
#endif
#pragma endregion

#pragma region CpyVal
            __host__ __device__ inline void CpyValIn(uint64_t Idx, const _T& Val) const {
#ifdef __CUDA_ARCH__
                memcpy(IdxToPtr(Idx), &Val, sizeof(_T));
#else
                ThrowIfBad(cudaMemcpy(IdxToPtr(Idx), &Val, sizeof(_T), cudaMemcpyHostToDevice));
#endif
            }
            __host__ __device__ inline void CpyValIn(const typename this_t::vector_t& Coords, const _T& Val) const {
#ifdef __CUDA_ARCH__
                memcpy(CoordsToPtr(Coords), &Val, sizeof(_T));
#else
                ThrowIfBad(cudaMemcpy(CoordsToPtr(Coords), &Val, sizeof(_T), cudaMemcpyHostToDevice));
#endif
            }
            template <bool _CopyFromHost>
            __host__ inline void CpyValIn(uint64_t Idx, const _T* Val) const {
                ThrowIfBad(cudaMemcpy(IdxToPtr(Idx), Val, sizeof(_T), _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
            }
#ifdef __CUDACC__
            __device__ inline void CpyValIn(uint64_t Idx, const _T* Val) const {
                memcpy(IdxToPtr(Idx), Val, sizeof(_T));
            }
#endif
            template <bool _CopyFromHost>
            __host__ inline void CpyValIn(const typename this_t::vector_t& Coords, const _T* Val) const {
                ThrowIfBad(cudaMemcpy(CoordsToPtr(Coords), Val, sizeof(_T), _CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
            }
#ifdef __CUDACC__
            __device__ inline void CpyValIn(const typename this_t::vector_t& Coords, const _T* Val) const {
                memcpy(CoordsToPtr(Coords), &Val, sizeof(_T));
            }
#endif
            __host__ __device__ inline void CpyValOut(uint64_t Idx, _T& Val) const {
#ifdef __CUDA_ARCH__
                memcpy(&Val, IdxToPtr(Idx), sizeof(_T));
#else
                ThrowIfBad(cudaMemcpy(&Val, IdxToPtr(Idx), sizeof(_T), cudaMemcpyDeviceToHost));
#endif
            }
            __host__ __device__ inline void CpyValOut(const typename this_t::vector_t& Coords, _T& Val) const {
#ifdef __CUDA_ARCH__
                memcpy(&Val, CoordsToPtr(Coords), sizeof(_T));
#else
                ThrowIfBad(cudaMemcpy(&Val, CoordsToPtr(Coords), sizeof(_T), cudaMemcpyDeviceToHost));
#endif
            }
            template <bool _CopyToHost>
            __host__ inline void CpyValOut(uint64_t Idx, _T* Val) const {
                ThrowIfBad(cudaMemcpy(Val, IdxToPtr(Idx), sizeof(_T), _CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
            }
#ifdef __CUDACC__
            __device__ inline void CpyValOut(uint64_t Idx, _T* Val) const {
                memcpy(Val, IdxToPtr(Idx), sizeof(_T));
            }
#endif
            template <bool _CopyToHost>
            __host__ inline void CpyValOut(const typename this_t::vector_t& Coords, _T* Val) const {
                ThrowIfBad(cudaMemcpy(Val, CoordsToPtr(Coords), sizeof(_T), _CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
            }
#ifdef __CUDACC__
            __device__ inline void CpyValOut(const typename this_t::vector_t& Coords, _T* Val) const {
                memcpy(Val, CoordsToPtr(Coords), sizeof(_T));
            }
#endif
            __host__ __device__ inline _T CpyValOut(uint64_t Idx) const {
                _T val;
                CpyValOut(Idx, val);
                return val;
            }
            __host__ __device__ inline _T CpyValOut(const typename this_t::vector_t& Coords) const {
                _T val;
                CpyValOut(Coords, val);
                return val;
            }
#pragma endregion

            __host__ __device__ inline void Dispose() {
#ifdef __CUDA_ARCH__
                free(darr);
#else
                ThrowIfBad(cudaFree(darr));
#endif
            }

            __host__ __device__ inline _T* Data() const {
                return darr;
            }

            template <bool _InputOnHost>
            __host__ inline void CopyBlockIn(const _T* Input, const typename this_t::vector_t& InputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                CopyBlock<_T, _DimensionCount, _InputOnHost, false, true>(Input, darr, InputDimensions, this->Dimensions(), RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ inline void CopyBlockIn(const _T* Input, const typename this_t::vector_t& InputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                CopyBlock<_T, _DimensionCount, true>(Input, darr, InputDimensions, this->Dimensions(), RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            template <bool _OutputOnHost>
            __host__ inline void CopyBlockOut(_T* Output, const typename this_t::vector_t& OutputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                CopyBlock<_T, _DimensionCount, false, _OutputOnHost, true>(darr, Output, this->Dimensions(), OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ inline void CopyBlockOut(_T* Output, const typename this_t::vector_t& OutputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                CopyBlock<_T, _DimensionCount, true>(darr, Output, this->Dimensions(), OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            
            __host__ __device__ inline this_t Clone() const {
                return FieldBase<_T, _DimensionCount>(this->Dimensions(), darr);
            }

            inline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                size_t t = sizeof(uint32_t) * _DimensionCount;
                size_t l = this->ValueCount();
                for (size_t i = 0; i < l; ++i)
                    t += BSerializer::SerializedSize(CpyValOut(i));
                return t;
            }
            inline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                BSerializer::Serialize<typename this_t::vector_t>(Data, this->Dimensions());
                size_t l = this->ValueCount();
                for (size_t i = 0; i < l; ++i)
                    BSerializer::Serialize(Data, CpyValOut(i));
            }
            static inline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                typename this_t::vector_t dimensions = BSerializer::Deserialize<typename this_t::vector_t>(Data);
                FieldBase<_T, _DimensionCount> field(dimensions);
                size_t l = field.ValueCount();
                for (size_t i = 0; i < l; ++i)
                    field.CpyValIn(i, BSerializer::Deserialize<_T>(Data));
                return field;
            }
            static inline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> requires BSerializer::Serializable<_T> {
                new (Value) FieldBase<_T, _DimensionCount>(Deserialize(Data));
            }
        private:
            _T* darr;
        };
    }
}