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
        class DField final : public details::DFieldBase<_T, _DimensionCount> {
            using this_t = DField<_T, _DimensionCount>;
            using base_t = details::DFieldBase<_T, _DimensionCount>;
            using vector_t = base_t::vector_t;
        public:
            using base_t::F;
            using base_t::B;
            using base_t::FData;
            using base_t::BData;
            using base_t::Reverse;
            using base_t::CpyAllIn;
            using base_t::CpyValIn;
            using base_t::CopyBlockIn;

            __host__ __device__ __forceinline this_t Clone() const {
                return *(this_t*)base_t::Clone();
            }
            static __forceinline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                return *(this_t*)base_t::Deserialize(Data);
            }
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
                base_t::Deserialize(Data, Value);
            }

            __host__ __device__ __forceinline DField(vector_t Dimensions)
                : base_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DField(_Ts... Dimensions)
                : DField(vector_t(Dimensions...)) { }
            __host__ __device__ __forceinline DField(vector_t Dimensions, _T* All)
                : base_t(Dimensions) {
                CpyAllIn(All);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DField(_Ts... Dimensions, _T* All)
                : DField(vector_t(Dimensions...), All) { }

            __host__ __device__ __forceinline DField(const this_t& Other)
                : DField(Other.Dimensions(), Other.FData()) { }
            __host__ __device__ __forceinline DField(this_t&& Other)
                : base_t(Other.Dimensions(), Other.FData(), Other.BData()) {
                new (&Other) base_t(this->Dimensions(), 0, 0);
            }
            __host__ __device__ __forceinline ~DField() {
                base_t::Dispose();
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

            __host__ __device__ __forceinline operator DFieldProxy<_T, _DimensionCount>() {
                return MakeProxy();
            }
            __host__ __device__ __forceinline operator DFieldProxyConst<_T, _DimensionCount>() const {
                return MakeProxy();
            }
        };
        template <typename _T, size_t _DimensionCount>
        class DFieldProxy final : public details::DFieldBase<_T, _DimensionCount> {
            using base_t = details::DFieldBase<_T, _DimensionCount>;
            using vector_t = base_t::vector_t;
        public:
            using base_t::F;
            using base_t::B;
            using base_t::FData;
            using base_t::BData;
            using base_t::Reverse;
            using base_t::CpyAllIn;
            using base_t::CpyValIn;
            using base_t::CopyBlockIn;

            __host__ __device__ __forceinline DField<_T, _DimensionCount> Clone() const {
                return *(DField<_T, _DimensionCount>*)base_t::Clone();
            }

            __host__ __device__ DFieldProxy(const vector_t& Dimensions, _T* ArrF, _T* ArrB)
                : base_t(Dimensions, ArrF, ArrB) { }
            __host__ __device__ DFieldProxy(DField<_T, _DimensionCount>& Parent)
                : base_t(Parent.Dimensions(), Parent.FData(), Parent.BData()) { }
            __host__ __device__ operator DFieldProxyConst<_T, _DimensionCount>() const {
                return DFieldProxyConst<_T, _DimensionCount>(*this);
            }
        };
        template <typename _T, size_t _DimensionCount>
        class DFieldProxyConst final : public details::DFieldBase<_T, _DimensionCount> {
            using base_t = details::DFieldBase<_T, _DimensionCount>;
            using vector_t = base_t::vector_t;
        public:
            __host__ __device__ __forceinline DField<_T, _DimensionCount> Clone() const {
                return *(DField<_T, _DimensionCount>*)base_t::Clone();
            }

            __host__ __device__ DFieldProxyConst(const vector_t& Dimensions, const _T* ArrF, const _T* ArrB)
                : base_t(Dimensions, const_cast<_T*>(ArrF), const_cast<_T*>(ArrB)) { }
            __host__ __device__ DFieldProxyConst(const DField<_T, _DimensionCount>& Parent)
                : base_t(Parent.Dimensions(), Parent.FData(), Parent.BData()) { }
            __host__ __device__ DFieldProxyConst(const DFieldProxy<_T, _DimensionCount>& Partner)
                : base_t(Partner.Dimensions(), Partner.FData(), Partner.BData()) { }
        };

        template <typename _T, size_t _DimensionCount>
        using dfieldIteratorKernel_t = void(*)(FixedVector<uint32_t, _DimensionCount> Pos, DFieldProxyConst<_T, _DimensionCount> Previous, _T& NextVal);
    }
}