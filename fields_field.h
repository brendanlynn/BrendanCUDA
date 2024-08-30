#pragma once

#include "details_fieldbase.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/device_ptr.h>

namespace BrendanCUDA {
    namespace Fields {
        template <typename _T, size_t _DimensionCount>
        class Field;
        template <typename _T, size_t _DimensionCount>
        class FieldProxy;
        template <typename _T, size_t _DimensionCount>
        class FieldProxyConst;

        template <typename _T, size_t _DimensionCount>
        class Field : private details::FieldBase<_T, _DimensionCount> {
            using this_t = Field<_T, _DimensionCount>;
            using basefb_t = details::FieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        protected:
            using vector_t = basedb_t::vector_t;
        public:
#pragma region Wrapper
            __host__ __device__ __forceinline Field(const vector_t& Dimensions)
                : basefb_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline Field(_Ts... Dimensions)
                : basefb_t(Dimensions...) { }

            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4) {
                return basedb_t::LengthX();
            }
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
                return basedb_t::LengthY();
            }
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
                return basedb_t::LengthZ();
            }
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4) {
                return basedb_t::LengthW();
            }
            __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
                return basedb_t::Length(Idx);
            }
            __host__ __device__ __forceinline vector_t Dimensions() const {
                return basedb_t::Dimensions();
            }
            __host__ __device__ __forceinline dim3 DimensionsD() const {
                return basedb_t::DimensionsD();
            }
            __host__ __device__ __forceinline size_t ValueCount() const {
                return basedb_t::ValueCount();
            }
            __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const {
                return basedb_t::IdxToCoords(Index);
            }
            __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const {
                return basedb_t::CoordsToIdx(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
                return basedb_t::CoordsToIdx(Coords...);
            }

            __host__ __device__ __forceinline size_t SizeOnGPU() const {
                return basefb_t::SizeOnGPU();
            }
            __host__ __device__ __forceinline _T* IdxToPtr(uint64_t Idx) {
                return basefb_t::IdxToPtr(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> IdxToRef(uint64_t Idx) {
                return basefb_t::IdxToRef(Idx);
            }
            __host__ __device__ __forceinline _T* CoordsToPtr(const vector_t& Coords) {
                return basefb_t::CoordsToPtr(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline _T* CoordsToPtr(_Ts... Coords) {
                return basefb_t::CoordsToPtr(Coords...);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> CoordsToRef(const vector_t& Coords) {
                return basefb_t::CoordsToRef(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> CoordsToRef(_Ts... Coords) {
                return basefb_t::CoordsToRef(Coords...);
            }
            __host__ __device__ __forceinline const _T* IdxToPtr(uint64_t Idx) const {
                return basefb_t::IdxToPtr(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> IdxToRef(uint64_t Idx) const {
                return basefb_t::IdxToRef(Idx);
            }
            __host__ __device__ __forceinline const _T* CoordsToPtr(const vector_t& Coords) const {
                return basefb_t::CoordsToPtr(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline const _T* CoordsToPtr(_Ts... Coords) const {
                return basefb_t::CoordsToPtr(Coords...);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> CoordsToRef(const vector_t& Coords) const {
                return basefb_t::CoordsToRef(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> CoordsToRef(_Ts... Coords) const {
                return basefb_t::CoordsToRef(Coords...);
            }
            __host__ __device__ __forceinline uint64_t PtrToIdx(const _T* Ptr) const {
                return basefb_t::PtrToIdx(Ptr);
            }
            __host__ __device__ __forceinline vector_t PtrToCoords(const _T* Ptr) const {
                return basefb_t::PtrToCoords(Ptr);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> PtrToRef(const _T* Ptr) const {
                return basefb_t::PtrToRef(Ptr);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> PtrToRef(const _T* Ptr) {
                return basefb_t::PtrToRef(Ptr);
            }
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __device__ __forceinline uint64_t RefToIdx(const _T& Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __device__ __forceinline vector_t RefToCoords(const _T& Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __host__ __forceinline const _T* RefToPtr(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __forceinline const _T* RefToPtr(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __device__ __forceinline const  _T* RefToPtr(const _T& Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __forceinline _T* RefToPtr(thrust::device_reference<_T> Ref) {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __forceinline _T* RefToPtr(thrust::device_reference<const _T> Ref) {
                return basefb_t::RefToPtr(Ref);
            }
            __device__ __forceinline _T* RefToPtr(const _T& Ref) {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> operator()(uint64_t Idx) const {
                return IdxToRef(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(uint64_t Idx) {
                return IdxToRef(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> operator()(const vector_t& Coords) const {
                return CoordsToRef(Coords);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(const vector_t& Coords) {
                return CoordsToRef(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> operator()(_Ts... Coords) const {
                return CoordsToRef(Coords...);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(_Ts... Coords) {
                return CoordsToRef(Coords...);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyAllIn(const _T* All) {
                basefb_t::CpyAllIn<_CopyFromHost>(All);
            }
            __device__ __forceinline void CpyAllIn(const _T* All) {
                basefb_t::CpyAllIn(All);
            }
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut<_CopyToHost>();
            }
            __device__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut();
            }
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut<_CopyToHost>(All);
            }
            __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut(All);
            }
            __host__ __device__ __forceinline void CpyValIn(uint64_t Idx, const _T& Val) {
                basefb_t::CpyValIn(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T& Val) {
                basefb_t::CpyValIn(Coords, Val);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) {
                basefb_t::CpyValIn<_CopyFromHost>(Idx, Val);
            }
            __device__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) {
                basefb_t::CpyValIn(Idx, Val);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) {
                basefb_t::CpyValIn<_CopyFromHost>(Coords, Val);
            }
            __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) {
                basefb_t::CpyValIn(Coords, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(uint64_t Idx, _T& Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(const vector_t& Coords, _T& Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const {
                return basefb_t::CpyValOut(Idx);
            }
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const {
                return basefb_t::CpyValOut(Coords);
            }
            __host__ __device__ __forceinline _T* Data() {
                return basefb_t::Data();
            }
            __host__ __device__ __forceinline const _T* Data() const {
                return basefb_t::Data();
            }
            template <bool _InputOnHost>
            __host__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) {
                basefb_t::CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __device__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) {
                basefb_t::CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline Field(const vector_t& Dimensions, const _T* All)
                : basefb_t(Dimensions) {
                CpyAllIn(All);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline Field(_Ts... Dimensions, const _T* All)
                : Field(vector_t(Dimensions...), All) { }

            static __forceinline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                return *(this_t)basefb_t::Deserialize(Data);
            }
            static __forceinline this_t Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
                basefb_t::Deserialize(Data, Value);
            }

            __host__ __device__ __forceinline Field(const this_t& Other)
                : Field(Other.Dimensions(), Other.Data()) { }
            __host__ __device__ __forceinline Field(this_t&& Other)
                : basefb_t(Other.Dimensions(), Other.Data()) {
                new (&Other) basefb_t(this->Dimensions(), 0);
            }
            __host__ __device__ __forceinline ~Field() {
                basefb_t::Dispose();
            }
            __host__ __device__ __forceinline this_t& operator=(const this_t& Other) {
                this->~Field();
                new (this) Field(Other);
                return *this;
            }
            __host__ __device__ __forceinline this_t& operator=(this_t&& Other) {
                this->~Field();
                new (this) Field(Other);
                return *this;
            }

            __host__ __device__ __forceinline FieldProxy<_T, _DimensionCount> MakeProxy() {
                return FieldProxy<_T, _DimensionCount>(*this);
            }
            __host__ __device__ __forceinline FieldProxyConst<_T, _DimensionCount> MakeConstProxy() const {
                return FieldProxyConst<_T, _DimensionCount>(*this);
            }
        };
        template <typename _T, size_t _DimensionCount>
        class FieldProxy : private details::FieldBase<_T, _DimensionCount> {
            using this_t = FieldProxy<_T, _DimensionCount>;
            using basefb_t = details::FieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        protected:
            using vector_t = basedb_t::vector_t;
        public:
#pragma region Wrapper
            __host__ __device__ __forceinline FieldProxy(const vector_t& Dimensions)
                : basefb_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline FieldProxy(_Ts... Dimensions)
                : basefb_t(Dimensions...) { }
            __host__ __device__ __forceinline FieldProxy(const vector_t& Dimensions, const _T* All)
                : basefb_t(Dimensions, All) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline FieldProxy(_Ts... Dimensions, const _T* All)
                : basefb_t(Dimensions..., All) { }

            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4) {
                return basedb_t::LengthX();
            }
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
                return basedb_t::LengthY();
            }
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
                return basedb_t::LengthZ();
            }
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4) {
                return basedb_t::LengthW();
            }
            __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
                return basedb_t::Length(Idx);
            }
            __host__ __device__ __forceinline vector_t Dimensions() const {
                return basedb_t::Dimensions();
            }
            __host__ __device__ __forceinline dim3 DimensionsD() const {
                return basedb_t::DimensionsD();
            }
            __host__ __device__ __forceinline size_t ValueCount() const {
                return basedb_t::ValueCount();
            }
            __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const {
                return basedb_t::IdxToCoords(Index);
            }
            __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const {
                return basedb_t::CoordsToIdx(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
                return basedb_t::CoordsToIdx(Coords...);
            }

            __host__ __device__ __forceinline size_t SizeOnGPU() const {
                return basefb_t::SizeOnGPU();
            }
            __host__ __device__ __forceinline _T* IdxToPtr(uint64_t Idx) const {
                return basefb_t::IdxToPtr(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> IdxToRef(uint64_t Idx) const {
                return basefb_t::IdxToRef(Idx);
            }
            __host__ __device__ __forceinline _T* CoordsToPtr(const vector_t& Coords) const {
                return basefb_t::CoordsToPtr(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline _T* CoordsToPtr(_Ts... Coords) const {
                return basefb_t::CoordsToPtr(Coords...);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> CoordsToRef(const vector_t& Coords) const {
                return basefb_t::CoordsToRef(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> CoordsToRef(_Ts... Coords) const {
                return basefb_t::CoordsToRef(Coords...);
            }
            __host__ __device__ __forceinline uint64_t PtrToIdx(const _T* Ptr) const {
                return basefb_t::PtrToIdx(Ptr);
            }
            __host__ __device__ __forceinline vector_t PtrToCoords(const _T* Ptr) const {
                return basefb_t::PtrToCoords(Ptr);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> PtrToRef(const _T* Ptr) const {
                return basefb_t::PtrToRef(Ptr);
            }
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __device__ __forceinline uint64_t RefToIdx(const _T& Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __device__ __forceinline vector_t RefToCoords(const _T& Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __host__ __forceinline _T* RefToPtr(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __forceinline _T* RefToPtr(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __device__ __forceinline _T* RefToPtr(const _T& Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(uint64_t Idx) const {
                return IdxToRef(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(const vector_t& Coords) const {
                return CoordsToRef(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(_Ts... Coords) const {
                return CoordsToRef(Coords...);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyAllIn(const _T* All) const {
                basefb_t::CpyAllIn<_CopyFromHost>(All);
            }
            __device__ __forceinline void CpyAllIn(const _T* All) const {
                basefb_t::CpyAllIn(All);
            }
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut<_CopyToHost>();
            }
            __device__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut();
            }
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut<_CopyToHost>(All);
            }
            __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut(All);
            }
            __host__ __device__ __forceinline void CpyValIn(uint64_t Idx, const _T& Val) const {
                basefb_t::CpyValIn(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T& Val) const {
                basefb_t::CpyValIn(Coords, Val);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val const) {
                basefb_t::CpyValIn<_CopyFromHost>(Idx, Val);
            }
            __device__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) const {
                basefb_t::CpyValIn(Idx, Val);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) const {
                basefb_t::CpyValIn<_CopyFromHost>(Coords, Val);
            }
            __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) const {
                basefb_t::CpyValIn(Coords, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(uint64_t Idx, _T& Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(const vector_t& Coords, _T& Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const {
                return basefb_t::CpyValOut(Idx);
            }
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const {
                return basefb_t::CpyValOut(Coords);
            }
            __host__ __device__ __forceinline _T* Data() const {
                return basefb_t::Data();
            }
            template <bool _InputOnHost>
            __host__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __device__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                basefb_t::Serialize(Data);
            }
#pragma endregion
            
            static __forceinline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                return *(this_t)basefb_t::Deserialize(Data);
            }
            static __forceinline this_t Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
                basefb_t::Deserialize(Data, Value);
            }

            __host__ __device__ __forceinline Field<_T, _DimensionCount> Clone() const {
                return *(Field<_T, _DimensionCount>*)basefb_t::Clone();
            }

            __host__ __device__ FieldProxy(Field<_T, _DimensionCount>& Parent)
                : FieldProxy(Parent.Dimensions(), Parent.Data()) { }

            __host__ __device__ __forceinline FieldProxyConst<_T, _DimensionCount> MakeConstProxy() const {
                return FieldProxyConst<_T, _DimensionCount>(*this);
            }
        };
        template <typename _T, size_t _DimensionCount>
        class FieldProxyConst : private details::FieldBase<_T, _DimensionCount> {
            using this_t = FieldProxyConst<_T, _DimensionCount>;
            using basefb_t = details::FieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        protected:
            using vector_t = basedb_t::vector_t;
        public:
#pragma region Wrapper
            __host__ __device__ __forceinline FieldProxyConst(const vector_t& Dimensions)
                : basefb_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline FieldProxyConst(_Ts... Dimensions)
                : basefb_t(Dimensions...) { }
            __host__ __device__ __forceinline FieldProxyConst(const vector_t& Dimensions, const _T* All)
                : basefb_t(Dimensions, All) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline FieldProxyConst(_Ts... Dimensions, const _T* All)
                : basefb_t(Dimensions..., All) { }

            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4) {
                return basedb_t::LengthX();
            }
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
                return basedb_t::LengthY();
            }
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
                return basedb_t::LengthZ();
            }
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4) {
                return basedb_t::LengthW();
            }
            __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
                return basedb_t::Length(Idx);
            }
            __host__ __device__ __forceinline vector_t Dimensions() const {
                return basedb_t::Dimensions();
            }
            __host__ __device__ __forceinline dim3 DimensionsD() const {
                return basedb_t::DimensionsD();
            }
            __host__ __device__ __forceinline size_t ValueCount() const {
                return basedb_t::ValueCount();
            }
            __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const {
                return basedb_t::IdxToCoords(Index);
            }
            __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const {
                return basedb_t::CoordsToIdx(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
                return basedb_t::CoordsToIdx(Coords...);
            }

            __host__ __device__ __forceinline size_t SizeOnGPU() const {
                return basefb_t::SizeOnGPU();
            }
            __host__ __device__ __forceinline const _T* IdxToPtr(uint64_t Idx) const {
                return basefb_t::IdxToPtr(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> IdxToRef(uint64_t Idx) const {
                return basefb_t::IdxToRef(Idx);
            }
            __host__ __device__ __forceinline const _T* CoordsToPtr(const vector_t& Coords) const {
                return basefb_t::CoordsToPtr(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline const _T* CoordsToPtr(_Ts... Coords) const {
                return basefb_t::CoordsToPtr(Coords...);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> CoordsToRef(const vector_t& Coords) const {
                return basefb_t::CoordsToRef(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> CoordsToRef(_Ts... Coords) const {
                return basefb_t::CoordsToRef(Coords...);
            }
            __host__ __device__ __forceinline uint64_t PtrToIdx(const _T* Ptr) const {
                return basefb_t::PtrToIdx(Ptr);
            }
            __host__ __device__ __forceinline vector_t PtrToCoords(const _T* Ptr) const {
                return basefb_t::PtrToCoords(Ptr);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> PtrToRef(const _T* Ptr) const {
                return basefb_t::PtrToRef(Ptr);
            }
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __host__ __forceinline uint64_t RefToIdx(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __device__ __forceinline uint64_t RefToIdx(const _T& Ref) const {
                return basefb_t::RefToIdx(Ref);
            }
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __host__ __forceinline vector_t RefToCoords(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __device__ __forceinline vector_t RefToCoords(const _T& Ref) const {
                return basefb_t::RefToCoords(Ref);
            }
            __host__ __forceinline const _T* RefToPtr(thrust::device_reference<_T> Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __forceinline const _T* RefToPtr(thrust::device_reference<const _T> Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __device__ __forceinline const  _T* RefToPtr(const _T& Ref) const {
                return basefb_t::RefToPtr(Ref);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> operator()(uint64_t Idx) const {
                return IdxToRef(Idx);
            }
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> operator()(const vector_t& Coords) const {
                return CoordsToRef(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, const _T&, thrust::device_reference<const _T>> operator()(_Ts... Coords) const {
                return CoordsToRef(Coords...);
            }
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut<_CopyToHost>();
            }
            __device__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut();
            }
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut<_CopyToHost>(All);
            }
            __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut(All);
            }
            __host__ __device__ __forceinline void CpyValOut(uint64_t Idx, _T& Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(const vector_t& Coords, _T& Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const {
                return basefb_t::CpyValOut(Idx);
            }
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const {
                return basefb_t::CpyValOut(Coords);
            }
            __host__ __device__ __forceinline const _T* Data() const {
                return basefb_t::Data();
            }
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline Field<_T, _DimensionCount> Clone() const {
                return *(Field<_T, _DimensionCount>*)base_t::Clone();
            }

            __host__ __device__ FieldProxyConst(const vector_t& Dimensions, const _T* All)
                : base_t(Dimensions, const_cast<_T*>(All)) { }
            __host__ __device__ FieldProxyConst(const Field<_T, _DimensionCount>& Parent)
                : base_t(Parent.Dimensions(), Parent.Data()) { }
            __host__ __device__ FieldProxyConst(const FieldProxy<_T, _DimensionCount>& Partner)
                : base_t(Partner.Dimensions(), Partner.Data()) { }
        };
    }
}