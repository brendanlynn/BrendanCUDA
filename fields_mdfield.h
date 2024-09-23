#pragma once

#include "fields_dfield.h"
#include "fields_mfield.h"
#include <array>

namespace bcuda {
    namespace details {
        template <size_t _DimensionCount, bool _Public, typename _T>
        using publicPrivateSelector_t = std::conditional_t<_Public, fields::FieldProxyConst<_T, _DimensionCount>, _T>;

        template <size_t _DimensionCount, typename _TTypes, typename _TPublics>
        struct MDFPPIK;
        template <size_t _DimensionCount, typename... _Ts, bool... _Publics>
            requires (sizeof...(_Ts) == sizeof...(_Publics))
        struct MDFPPIK<_DimensionCount, std::tuple<_Ts...>, std::integer_sequence<bool, _Publics...>> {
            static constexpr size_t size = sizeof...(_Ts);
            static constexpr std::array<bool, size> pubArr{ _Publics... };

            template <uintmax_t _Idx>
            using idx_type_t = std::tuple_element_t<_Idx, std::tuple<_Ts...>>;
            template <uintmax_t _Idx>
            static constexpr bool idx_val = pubArr[_Idx];

            template <uintmax_t _Idx>
            using idx_fieldType_t = publicPrivateSelector_t<_DimensionCount, idx_val<_Idx>, idx_type_t<_Idx>>;
        
            template <typename _TIndicies>
            struct Type2;
            template <size_t... _Idxs>
            struct Type2<std::index_sequence<_Idxs...>> {
                using type_t = void(*)(const FixedVector<uint32_t, _DimensionCount>& Pos, const idx_fieldType_t<_Idxs>&... Prevs, idx_type_t<_Idxs>&... Next);
            };

            using type_t = Type2<std::make_index_sequence<size>>::type_t;
        };

        template <size_t _DimensionCount, typename _TTypes, typename _TPublics>
        using mdfppik_t = typename MDFPPIK<_DimensionCount, _TTypes, _TPublics>::type_t;

        template <size_t _DimensionCount, typename... _Ts>
        using mdfkf_t = void(*)(const FixedVector<uint32_t, _DimensionCount>& Pos, const fields::FieldProxyConst<_Ts, _DimensionCount>&... Prevs, _Ts&... Next);
    }

    namespace fields {
        template <size_t _DimensionCount, typename... _Ts>
        class MDField;
        template <size_t _DimensionCount, typename... _Ts>
        class MDFieldProxy;
        template <size_t _DimensionCount, typename... _Ts>
        class MDFieldProxyConst;

        template <size_t _DimensionCount, typename... _Ts>
        class MDField {
            using this_t = MDField<_DimensionCount, _Ts...>;
            using basemf_t = MField<_DimensionCount, _Ts..., _Ts...>;
            basemf_t fields;
        public:
            using vector_t = basemf_t::vector_t;
            using tuple_t = std::tuple<_Ts...>;
            template <size_t _Idx>
            using element_t = std::tuple_element_t<_Idx, tuple_t>;
            using kernelFunc_t = details::mdfkf_t<_DimensionCount, _Ts...>;
            template <bool... _Publics>
                requires (sizeof...(_Publics) == sizeof...(_Ts))
            using publicPrivateKernelFunc_t = details::mdfppik_t<_DimensionCount, std::tuple<_Ts...>, std::integer_sequence<bool, _Publics...>>;

#pragma region Wrapper
            __host__ __device__ __forceinline MDField(const vector_t& Dimensions)
                : fields(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline MDField(_Ts... Dimensions)
                : fields(vector_t(Dimensions...)) { }

            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4) {
                return fields.LengthX();
            }
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
                return fields.LengthY();
            }
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
                return fields.LengthZ();
            }
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4) {
                return fields.LengthW();
            }
            __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
                return fields.Length(Idx);
            }
            __host__ __device__ __forceinline vector_t Dimensions() const {
                return fields.Dimensions();
            }
            __host__ __device__ __forceinline dim3 DimensionsD() const {
                return fields.DimensionsD();
            }
            __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const {
                return fields.IdxToCoords(Index);
            }
            __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const {
                return fields.CoordsToIdx(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
                return fields.CoordsToIdx(Coords...);
            }

            __host__ __device__ __forceinline size_t EachValueCount() const {
                return fields.EachValueCount();
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ __forceinline size_t EachSizeOnGPU() const {
                return fields.EachSizeOnGPU<_Idx>();
            }
            __host__ __device__ __forceinline size_t TotalValueCount() const {
                return fields.TotalValueCount();
            }
            __host__ __device__ __forceinline size_t TotalSizeOnGPU() const {
                return fields.TotalSizeOnGPU();
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ FieldProxy<element_t<_Idx>, _DimensionCount> F() {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.F<tIdx>();
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ FieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.FConst<tIdx>();
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ DFieldProxy<element_t<_Idx>, _DimensionCount> F() {
                return DFieldProxy<element_t<_Idx>, _DimensionCount>(Dimensions(), fields.FData<_Idx>(), fields.FData<_Idx + sizeof...(_Ts)>());
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ DFieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                return DFieldProxyConst<element_t<_Idx>, _DimensionCount>(Dimensions(), fields.FData<_Idx>(), fields.FData<_Idx + sizeof...(_Ts)>());
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ element_t<_Idx>* FData() {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.FData<tIdx>();
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ const element_t<_Idx>* FData() const {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.FData<tIdx>();
            }
            __forceinline size_t SerializedSize() const requires (BSerializer::Serializable<_Ts> && ...) {
                return fields.SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires (BSerializer::Serializable<_Ts> && ...) {
                fields.Serialize(Data);
            }
#pragma endregion

            static __forceinline this_t Deserialize(const void*& Data) requires (BSerializer::Serializable<_Ts> && ...) {
                return *(this_t*)&details::MFieldBase<_DimensionCount, _Ts..., _Ts...>(Data);
            }
            static __forceinline void Deserialize(const void*& Data, void* Value) requires (BSerializer::Serializable<_Ts> && ...) {
                details::MFieldBase<_DimensionCount, _Ts..., _Ts...>(Data, Value);
            }

            __forceinline MDField(const this_t&) = default;
            __forceinline MDField(this_t&&) = default;
            __forceinline this_t& operator=(const this_t&) = default;
            __forceinline this_t& operator=(this_t&&) = default;

            __host__ __device__ __forceinline MDFieldProxy<_DimensionCount, _Ts...> MakeProxy() {
                return MDFieldProxy<_DimensionCount, _Ts...>(*this);
            }
            __host__ __device__ __forceinline MDFieldProxyConst<_DimensionCount, _Ts...> MakeProxyConst() const {
                return MDFieldProxyConst<_DimensionCount, _Ts...>(*this);
            }

            __host__ __device__ __forceinline void Reverse() {
                void* const* oldArrs = ((details::MFieldBase<_DimensionCount, _Ts..., _Ts...>*)&fields)->FieldDataArray();
                void* arrs[sizeof...(_Ts) << 1];

                for (size_t i = 0; i < sizeof...(_Ts); ++i) {
                    arrs[i] = oldArrs[i + sizeof...(_Ts)];
                    arrs[i + sizeof...(_Ts)] = oldArrs[i];
                }

                vector_t dims = Dimensions();

                new (this) MDFieldProxy<_DimensionCount, _Ts...>(dims, (void* const*)&arrs);
            }
        };
        template <size_t _DimensionCount, typename... _Ts>
        class MDFieldProxy {
            using this_t = MDFieldProxy<_DimensionCount, _Ts...>;
            using basemf_t = MFieldProxy<_DimensionCount, _Ts..., _Ts...>;
            basemf_t fields;
        public:
            using vector_t = basemf_t::vector_t;
            using tuple_t = std::tuple<_Ts...>;
            template <size_t _Idx>
            using element_t = std::tuple_element_t<_Idx, tuple_t>;
            using kernelFunc_t = details::mdfkf_t<_DimensionCount, _Ts...>;
            template <bool... _Publics>
                requires (sizeof...(_Publics) == sizeof...(_Ts))
            using publicPrivateKernelFunc_t = details::mdfppik_t<_DimensionCount, std::tuple<_Ts...>, std::integer_sequence<bool, _Publics...>>;

#pragma region Wrapper
            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4) {
                return fields.LengthX();
            }
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
                return fields.LengthY();
            }
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
                return fields.LengthZ();
            }
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4) {
                return fields.LengthW();
            }
            __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
                return fields.Length(Idx);
            }
            __host__ __device__ __forceinline vector_t Dimensions() const {
                return fields.Dimensions();
            }
            __host__ __device__ __forceinline dim3 DimensionsD() const {
                return fields.DimensionsD();
            }
            __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const {
                return fields.IdxToCoords(Index);
            }
            __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const {
                return fields.CoordsToIdx(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
                return fields.CoordsToIdx(Coords...);
            }

            __host__ __device__ __forceinline size_t EachValueCount() const {
                return fields.EachValueCount();
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ __forceinline size_t EachSizeOnGPU() const {
                return fields.EachSizeOnGPU<_Idx>();
            }
            __host__ __device__ __forceinline size_t TotalValueCount() const {
                return fields.TotalValueCount();
            }
            __host__ __device__ __forceinline size_t TotalSizeOnGPU() const {
                return fields.TotalSizeOnGPU();
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ FieldProxy<element_t<_Idx>, _DimensionCount> F() const {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.F<tIdx>();
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ FieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.FConst<tIdx>();
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ DFieldProxy<element_t<_Idx>, _DimensionCount> F() const {
                return DFieldProxy<element_t<_Idx>, _DimensionCount>(Dimensions(), const_cast<element_t<_Idx>*>(fields.FData<_Idx>()), const_cast<element_t<_Idx>*>(fields.FData<_Idx + sizeof...(_Ts)>()));
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ DFieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                return DFieldProxyConst<element_t<_Idx>, _DimensionCount>(Dimensions(), fields.FData<_Idx>(), fields.FData<_Idx + sizeof...(_Ts)>());
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ element_t<_Idx>* FData() const {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.FData<tIdx>();
            }
            __forceinline size_t SerializedSize() const requires (BSerializer::Serializable<_Ts> && ...) {
                return fields.SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires (BSerializer::Serializable<_Ts> && ...) {
                fields.Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline MDField<_DimensionCount, _Ts...> Clone() const {
                return *(MDField<_DimensionCount, _Ts...>*)&((details::MFieldBase<_DimensionCount, _Ts..., _Ts...>)&fields)->Clone();
            }

            __host__ __device__ MDFieldProxy(const vector_t& Dimensions, void* const* Arrays)
                : fields(Dimensions, Arrays) { }
            __host__ __device__ MDFieldProxy(MDField<_DimensionCount, _Ts...>& Parent)
                : fields(Parent.Dimensions(), ((details::MFieldBase<_DimensionCount, _Ts..., _Ts...>*)&(((this_t*)&Parent)->fields))->FieldDataArray()) { }
        };
        template <size_t _DimensionCount, typename... _Ts>
        class MDFieldProxyConst {
            using this_t = MDFieldProxyConst<_DimensionCount, _Ts...>;
            using basemf_t = MFieldProxyConst<_DimensionCount, _Ts..., _Ts...>;
            basemf_t fields;
        public:
            using vector_t = basemf_t::vector_t;
            using tuple_t = std::tuple<_Ts...>;
            template <size_t _Idx>
            using element_t = std::tuple_element_t<_Idx, tuple_t>;

#pragma region Wrapper
            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4) {
                return fields.LengthX();
            }
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
                return fields.LengthY();
            }
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
                return fields.LengthZ();
            }
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4) {
                return fields.LengthW();
            }
            __host__ __device__ __forceinline uint32_t Length(size_t Idx) const {
                return fields.Length(Idx);
            }
            __host__ __device__ __forceinline vector_t Dimensions() const {
                return fields.Dimensions();
            }
            __host__ __device__ __forceinline dim3 DimensionsD() const {
                return fields.DimensionsD();
            }
            __host__ __device__ __forceinline vector_t IdxToCoords(uint64_t Index) const {
                return fields.IdxToCoords(Index);
            }
            __host__ __device__ __forceinline uint64_t CoordsToIdx(vector_t Coords) const {
                return fields.CoordsToIdx(Coords);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordsToIdx(_Ts... Coords) const {
                return fields.CoordsToIdx(Coords...);
            }

            __host__ __device__ __forceinline size_t EachValueCount() const {
                return fields.EachValueCount();
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ __forceinline size_t EachSizeOnGPU() const {
                return fields.EachSizeOnGPU<_Idx>();
            }
            __host__ __device__ __forceinline size_t TotalValueCount() const {
                return fields.TotalValueCount();
            }
            __host__ __device__ __forceinline size_t TotalSizeOnGPU() const {
                return fields.TotalSizeOnGPU();
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ FieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.FConst<tIdx>();
            }
            template <size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ DFieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                return DFieldProxyConst<element_t<_Idx>, _DimensionCount>(Dimensions(), fields.FData<_Idx>(), fields.FData<_Idx + sizeof...(_Ts)>());
            }
            template <bool _Front, size_t _Idx>
                requires (_Idx < sizeof...(_Ts))
            __host__ __device__ const element_t<_Idx>* FData() const {
                constexpr size_t tIdx = _Front ? _Idx : sizeof...(_Ts) + _Idx;
                return fields.FData<tIdx>();
            }
            __forceinline size_t SerializedSize() const requires (BSerializer::Serializable<_Ts> && ...) {
                return fields.SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires (BSerializer::Serializable<_Ts> && ...) {
                fields.Serialize(Data);
            }
#pragma endregion

            __host__ __device__ __forceinline MDField<_DimensionCount, _Ts...> Clone() const {
                return *(MDField<_DimensionCount, _Ts...>*)&(((details::MFieldBase<_DimensionCount, _Ts..., _Ts...>*)&fields)->Clone());
            }

            __host__ __device__ MDFieldProxyConst(const vector_t& Dimensions, void* const* Arrays)
                : fields(Dimensions, Arrays) { }
            __host__ __device__ MDFieldProxyConst(MDField<_DimensionCount, _Ts...>& Parent)
                : fields(Parent.Dimensions(), &((this_t*)&Parent)->fields) { }
            __host__ __device__ MDFieldProxyConst(MDFieldProxy<_DimensionCount, _Ts...>& Partner)
                : fields(Partner.Dimensions(), &((this_t*)&Partner)->fields) { }
        };
    }
}