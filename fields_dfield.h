#pragma once

#include "errorhelp.h"
#include "details_dfieldbase.h"
#include "points.h"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <typename _T, size_t _DimensionCount>
        class DField;
        template <typename _T, size_t _DimensionCount>
        class DFieldProxy;
        template <typename _T, size_t _DimensionCount>
        class DFieldProxyConst;

        template <typename _T, size_t _DimensionCount>
        class DField : private details::DFieldBase<_T, _DimensionCount> {
            using this_t = DField<_T, _DimensionCount>;
            using basefb_t = details::DFieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        protected:
            using vector_t = basefb_t::vector_t;
        public:
#pragma region Wrapper
            __host__ __device__ __forceinline DField(const vector_t& Dimensions)
                : basefb_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DField(_Ts... Dimensions)
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
            __host__ __device__ Fields::FieldProxy<_T, _DimensionCount> F() {
                return basefb_t::F();
            }
            __host__ __device__ Fields::FieldProxy<_T, _DimensionCount> B() {
                return basefb_t::B();
            }
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> FConst() const {
                return basefb_t::FConst();
            }
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> BConst() const {
                return basefb_t::BConst();
            }
            __host__ __device__ _T* FData() {
                return basefb_t::FData();
            }
            __host__ __device__ _T* BData() {
                return basefb_t::BData();
            }
            __host__ __device__ const _T* FData() const {
                return basefb_t::FData();
            }
            __host__ __device__ const _T* BData() const {
                return basefb_t::BData();
            }
            __host__ __device__ void Reverse() {
                basefb_t::Reverse();
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyAllIn(const _T* All) {
                basefb_t::template CpyAllIn<_CopyFromHost>(All);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllIn(const _T* All) {
                basefb_t::CpyAllIn(All);
            }
#endif
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const {
                return basefb_t::template CpyAllOut<_CopyToHost>();
            }
#ifdef __CUDACC__
            __device__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut();
            }
#endif
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::template CpyAllOut<_CopyToHost>(All);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut(All);
            }
#endif
            __host__ __device__ __forceinline void CpyValIn(uint64_t Idx, const _T& Val) {
                basefb_t::CpyValIn(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T& Val) {
                basefb_t::CpyValIn(Coords, Val);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) {
                basefb_t::template CpyValIn<_CopyFromHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) {
                basefb_t::CpyValIn(Idx, Val);
            }
#endif
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) {
                basefb_t::template CpyValIn<_CopyFromHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) {
                basefb_t::CpyValIn(Coords, Val);
            }
#endif
            __host__ __device__ __forceinline void CpyValOut(uint64_t Idx, _T& Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(const vector_t& Coords, _T& Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::template CpyValOut<_CopyToHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
#endif
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::template CpyValOut<_CopyToHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
#endif
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const {
                return basefb_t::CpyValOut(Idx);
            }
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const {
                return basefb_t::CpyValOut(Coords);
            }
            template <bool _InputOnHost>
            __host__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) {
                basefb_t::template CopyBlockIn<_InputOnHost>(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) {
                basefb_t::CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::template CopyBlockOut<_OutputOnHost>(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            static __forceinline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                return *(this_t*)&basefb_t::Deserialize(Data);
            }
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
                basefb_t::Deserialize(Data, Value);
            }

            __host__ __device__ __forceinline DField(vector_t Dimensions, _T* All)
                : DField(Dimensions) {
                CpyAllIn(All);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DField(_Ts... Dimensions, _T* All)
                : DField(vector_t(Dimensions...), All) { }

            __host__ __device__ __forceinline DField(const this_t& Other)
                : DField(Other.Dimensions(), Other.FData()) { }
            __host__ __device__ __forceinline DField(this_t&& Other)
                : basefb_t(Other.Dimensions(), Other.FData(), Other.BData()) {
                new (&Other) basefb_t(this->Dimensions(), 0, 0);
            }
            __host__ __device__ __forceinline ~DField() {
                basefb_t::Dispose();
            }
            __host__ __device__ __forceinline this_t& operator=(const this_t& Other) {
                this->~DField();
                new (this) DField(Other);
                return *this;
            }
            __host__ __device__ __forceinline this_t& operator=(this_t&& Other) {
                this->~DField();
                new (this) DField(Other);
                return *this;
            }

            __host__ __device__ __forceinline DFieldProxy<_T, _DimensionCount> MakeProxy() {
                return DFieldProxy<_T, _DimensionCount>(*this);
            }
            __host__ __device__ __forceinline DFieldProxyConst<_T, _DimensionCount> MakeProxy() const {
                return DFieldProxyConst<_T, _DimensionCount>(*this);
            }
        };
        template <typename _T, size_t _DimensionCount>
        class DFieldProxy : private details::DFieldBase<_T, _DimensionCount> {
            using basefb_t = details::DFieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        protected:
            using vector_t = basefb_t::vector_t;
        public:
#pragma region Wrapper
            __host__ __device__ __forceinline DFieldProxy(const vector_t& Dimensions)
                : basefb_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DFieldProxy(_Ts... Dimensions)
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
            __host__ __device__ Fields::FieldProxy<_T, _DimensionCount> F() const {
                return basefb_t::F();
            }
            __host__ __device__ Fields::FieldProxy<_T, _DimensionCount> B() const {
                return basefb_t::B();
            }
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> FConst() const {
                return basefb_t::FConst();
            }
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> BConst() const {
                return basefb_t::BConst();
            }
            __host__ __device__ _T* FData() const {
                return basefb_t::FData();
            }
            __host__ __device__ _T* BData() const {
                return basefb_t::BData();
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyAllIn(const _T* All) const {
                basefb_t::template CpyAllIn<_CopyFromHost>(All);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllIn(const _T* All) const {
                basefb_t::CpyAllIn(All);
            }
#endif
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const {
                return basefb_t::template CpyAllOut<_CopyToHost>();
            }
#ifdef __CUDACC__
            __device__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut();
            }
#endif
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::template CpyAllOut<_CopyToHost>(All);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut(All);
            }
#endif
            __host__ __device__ __forceinline void CpyValIn(uint64_t Idx, const _T& Val) const {
                basefb_t::CpyValIn(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T& Val) const {
                basefb_t::CpyValIn(Coords, Val);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) const {
                basefb_t::template CpyValIn<_CopyFromHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) const {
                basefb_t::CpyValIn(Idx, Val);
            }
#endif
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) const {
                basefb_t::template CpyValIn<_CopyFromHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) const {
                basefb_t::CpyValIn(Coords, Val);
            }
#endif
            __host__ __device__ __forceinline void CpyValOut(uint64_t Idx, _T& Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(const vector_t& Coords, _T& Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::template CpyValOut<_CopyToHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
#endif
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::template CpyValOut<_CopyToHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
#endif
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const {
                return basefb_t::CpyValOut(Idx);
            }
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const {
                return basefb_t::CpyValOut(Coords);
            }
            template <bool _InputOnHost>
            __host__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::template CopyBlockIn<_InputOnHost>(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::template CopyBlockOut<_OutputOnHost>(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline DField<_T, _DimensionCount> Clone() const {
                return *(DField<_T, _DimensionCount>*)basefb_t::Clone();
            }

            __host__ __device__ DFieldProxy(const vector_t& Dimensions, _T* ArrF, _T* ArrB)
                : basefb_t(Dimensions, ArrF, ArrB) { }
            __host__ __device__ DFieldProxy(DField<_T, _DimensionCount>& Parent)
                : basefb_t(Parent.Dimensions(), Parent.FData(), Parent.BData()) { }
        };
        template <typename _T, size_t _DimensionCount>
        class DFieldProxyConst : private details::DFieldBase<_T, _DimensionCount> {
            using basefb_t = details::DFieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        protected:
            using vector_t = basefb_t::vector_t;
        public:
#pragma region Wrapper
            __host__ __device__ __forceinline DFieldProxyConst(const vector_t& Dimensions)
                : basefb_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DFieldProxyConst(_Ts... Dimensions)
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
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> FConst() const {
                return basefb_t::FConst();
            }
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> BConst() const {
                return basefb_t::BConst();
            }
            __host__ __device__ const _T* FData() const {
                return basefb_t::FData();
            }
            __host__ __device__ const _T* BData() const {
                return basefb_t::BData();
            }
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const {
                return basefb_t::template CpyAllOut<_CopyToHost>();
            }
#ifdef __CUDACC__
            __device__ __forceinline _T* CpyAllOut() const {
                return basefb_t::CpyAllOut();
            }
#endif
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::template CpyAllOut<_CopyToHost>(All);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllOut(_T* All) const {
                basefb_t::CpyAllOut(All);
            }
#endif
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
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                basefb_t::CpyValOut(Idx, Val);
            }
#endif
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                basefb_t::CpyValOut(Coords, Val);
            }
#endif
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const {
                return basefb_t::CpyValOut(Idx);
            }
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const {
                return basefb_t::CpyValOut(Coords);
            }
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::template CopyBlockOut<_OutputOnHost>(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                basefb_t::CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline DField<_T, _DimensionCount> Clone() const {
                return *(DField<_T, _DimensionCount>*)basefb_t::Clone();
            }

            __host__ __device__ DFieldProxyConst(const vector_t& Dimensions, const _T* ArrF, const _T* ArrB)
                : basefb_t(Dimensions, const_cast<_T*>(ArrF), const_cast<_T*>(ArrB)) { }
            __host__ __device__ DFieldProxyConst(const DField<_T, _DimensionCount>& Parent)
                : basefb_t(Parent.Dimensions(), Parent.FData(), Parent.BData()) { }
            __host__ __device__ DFieldProxyConst(const DFieldProxy<_T, _DimensionCount>& Partner)
                : basefb_t(Partner.Dimensions(), Partner.FData(), Partner.BData()) { }
        };

        template <typename _T, size_t _DimensionCount>
        using dfieldIteratorKernel_t = void(*)(FixedVector<uint32_t, _DimensionCount> Pos, FieldProxyConst<_T, _DimensionCount> Previous, _T& NextVal);
    }
}