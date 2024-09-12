#pragma once

#include "errorhelp.h"
#include "details_mfieldbase.h"
#include "fields_field.h"
#include "points.h"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <size_t _DimensionCount, typename... _Ts>
        class MField;
        template <size_t _DimensionCount, typename... _Ts>
        class MFieldProxy;
        template <size_t _DimensionCount, typename... _Ts>
        class MFieldProxyConst;

        template <size_t _DimensionCount, typename... _Ts>
        class MField : private details::MFieldBase<_DimensionCount, _Ts...> {
            using this_t = MField<_DimensionCount, _Ts...>;
            using basefb_t = details::MFieldBase<_DimensionCount, _Ts...>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        public:
            using vector_t = basefb_t::vector_t;
            using tuple_t = basefb_t::tuple_t;
            template <size_t _Idx>
            using element_t = basefb_t::template element_t<_Idx>;

#pragma region Wrapper
            __host__ __device__ __forceinline MField(const vector_t& Dimensions)
                : basefb_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline MField(_Ts... Dimensions)
                : basefb_t(vector_t(Dimensions...)) { }

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

            __host__ __device__ __forceinline size_t EachValueCount() const {
                return basefb_t::ValueCount();
            }
            template <size_t _Idx>
            __host__ __device__ __forceinline size_t EachSizeOnGPU() const {
                return basefb_t::template EachSizeOnGPU<_Idx>();
            }
            __host__ __device__ __forceinline size_t TotalValueCount() const {
                return basefb_t::TotalValueCount();
            }
            __host__ __device__ __forceinline size_t TotalSizeOnGPU() const {
                return basefb_t::TotalSizeOnGPU();
            }
            template <size_t _Idx>
            __host__ __device__ Fields::FieldProxy<this_t::template element_t<_Idx>, _DimensionCount> F() {
                return basefb_t::template F<_Idx>();
            }
            template <size_t _Idx>
            __host__ __device__ Fields::FieldProxyConst<this_t::template element_t<_Idx>, _DimensionCount> FConst() const {
                return basefb_t::template FConst<_Idx>();
            }
            template <size_t _Idx>
            __host__ __device__ this_t::template element_t<_Idx>* FData() {
                return basefb_t::template FData<_Idx>();
            }
            template <size_t _Idx>
            __host__ __device__ const this_t::template element_t<_Idx>* FData() const {
                return basefb_t::template FData<_Idx>();
            }
            __forceinline size_t SerializedSize() const requires (BSerializer::Serializable<_Ts> && ...) {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires (BSerializer::Serializable<_Ts> && ...) {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            static __forceinline this_t Deserialize(const void*& Data) requires (BSerializer::Serializable<_Ts> && ...) {
                return *(this_t*)&basefb_t::Deserialize(Data);
            }
            static __forceinline void Deserialize(const void*& Data, void* Value) requires (BSerializer::Serializable<_Ts> && ...) {
                basefb_t::Deserialize(Data, Value);
            }

            __host__ __device__ __forceinline MField(const this_t& Other)
                : MField(Other.Clone()) { }
            __host__ __device__ __forceinline MField(this_t&& Other)
                : basefb_t(Other.Dimensions(), Other.FieldDataArray()) {
                void* arrs[sizeof...(_Ts)];
                for (size_t i = 0; i < sizeof...(_Ts); ++i)
                    arrs[i] = 0;
                new (&Other) basefb_t(this->Dimensions(), &arrs);
            }
            __host__ __device__ __forceinline ~MField() {
                basefb_t::Dispose();
            }
            __host__ __device__ __forceinline this_t& operator=(const this_t& Other) {
                this->~MField();
                new (this) MField(Other);
                return *this;
            }
            __host__ __device__ __forceinline this_t& operator=(this_t&& Other) {
                this->~MField();
                new (this) MField(Other);
                return *this;
            }

            __host__ __device__ __forceinline MFieldProxy<_DimensionCount, _Ts...> MakeProxy() {
                return MFieldProxy<_DimensionCount, _Ts...>(*this);
            }
            __host__ __device__ __forceinline MFieldProxyConst<_DimensionCount, _Ts...> MakeProxyConst() const {
                return MFieldProxyConst<_DimensionCount, _Ts...>(*this);
            }
        };
        template <size_t _DimensionCount, typename... _Ts>
        class MFieldProxy : private details::MFieldBase<_DimensionCount, _Ts...> {
            using basefb_t = details::MFieldBase<_DimensionCount, _Ts...>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        public:
            using vector_t = basefb_t::vector_t;
            using tuple_t = basefb_t::tuple_t;
            template <size_t _Idx>
            using element_t = basefb_t::template element_t<_Idx>;

#pragma region Wrapper
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

            __host__ __device__ __forceinline size_t EachValueCount() const {
                return basefb_t::ValueCount();
            }
            template <size_t _Idx>
            __host__ __device__ __forceinline size_t EachSizeOnGPU() const {
                return basefb_t::template EachSizeOnGPU<_Idx>();
            }
            __host__ __device__ __forceinline size_t TotalValueCount() const {
                return basefb_t::TotalValueCount();
            }
            __host__ __device__ __forceinline size_t TotalSizeOnGPU() const {
                return basefb_t::TotalSizeOnGPU();
            }
            template <size_t _Idx>
            __host__ __device__ Fields::FieldProxy<element_t<_Idx>, _DimensionCount> F() const {
                return basefb_t::template F<_Idx>();
            }
            template <size_t _Idx>
            __host__ __device__ Fields::FieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                return basefb_t::template FConst<_Idx>();
            }
            template <size_t _Idx>
            __host__ __device__ element_t<_Idx>* FData() const {
                return basefb_t::template FData<_Idx>();
            }
            __forceinline size_t SerializedSize() const requires (BSerializer::Serializable<_Ts> && ...) {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires (BSerializer::Serializable<_Ts> && ...) {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline MField<_DimensionCount, _Ts...> Clone() const {
                return *(MField<_DimensionCount, _Ts...>*)&basefb_t::Clone();
            }

            __host__ __device__ MFieldProxy(const vector_t& Dimensions, void* const* Arrays)
                : basefb_t(Dimensions, Arrays) { }
            __host__ __device__ MFieldProxy(MField<_DimensionCount, _Ts...>& Parent)
                : basefb_t(Parent.Dimensions(), Parent.FieldDataArray()) { }
        };
        template <size_t _DimensionCount, typename... _Ts>
        class MFieldProxyConst : private details::MFieldBase<_DimensionCount, _Ts...> {
            using basefb_t = details::MFieldBase<_DimensionCount, _Ts...>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        public:
            using vector_t = basefb_t::vector_t;
            using tuple_t = basefb_t::tuple_t;
            template <size_t _Idx>
            using element_t = basefb_t::template element_t<_Idx>;

#pragma region Wrapper
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

            __host__ __device__ __forceinline size_t EachValueCount() const {
                return basefb_t::ValueCount();
            }
            template <size_t _Idx>
            __host__ __device__ __forceinline size_t EachSizeOnGPU() const {
                return basefb_t::template EachSizeOnGPU<_Idx>();
            }
            __host__ __device__ __forceinline size_t TotalValueCount() const {
                return basefb_t::TotalValueCount();
            }
            __host__ __device__ __forceinline size_t TotalSizeOnGPU() const {
                return basefb_t::TotalSizeOnGPU();
            }
            template <size_t _Idx>
            __host__ __device__ Fields::FieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                return basefb_t::template FConst<_Idx>();
            }
            template <size_t _Idx>
            __host__ __device__ const element_t<_Idx>* FData() const {
                return basefb_t::template FData<_Idx>();
            }
            __forceinline size_t SerializedSize() const requires (BSerializer::Serializable<_Ts> && ...) {
                return basefb_t::SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires (BSerializer::Serializable<_Ts> && ...) {
                basefb_t::Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline MField<_DimensionCount, _Ts...> Clone() const {
                return *(MField<_DimensionCount, _Ts...>*)&basefb_t::Clone();
            }

            __host__ __device__ MFieldProxyConst(const vector_t& Dimensions, const void* const* Arrays)
                : basefb_t(Dimensions, const_cast<void* const*>(Arrays)) { }
            __host__ __device__ MFieldProxyConst(const MField<_DimensionCount, _Ts...>& Parent)
                : basefb_t(Parent.Dimensions(), Parent.FieldDataArray()) { }
            __host__ __device__ MFieldProxyConst(const MFieldProxy<_DimensionCount, _Ts...>& Partner)
                : basefb_t(Partner.Dimensions(), Partner.FieldDataArray()) { }
        };
    }
}