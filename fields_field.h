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
        class Field final : public details::FieldBase<_T, _DimensionCount> {
            using this_t = Field<_T, _DimensionCount>;
            using base_t = details::FieldBase<_T, _DimensionCount>;
            using vector_t = base_t::vector_t;
        public:
            using base_t::IdxToPtr;
            using base_t::IdxToRef;
            using base_t::CoordsToPtr;
            using base_t::CoordsToRef;
            using base_t::PtrToRef;
            using base_t::RefToPtr;
            using base_t::operator();
            using base_t::CpyAllIn;
            using base_t::CpyValIn;
            using base_t::CopyBlockIn;
            using base_t::Data;

            __host__ __device__ __forceinline this_t Clone() const {
                return *(this_t*)base_t::Clone();
            }
            static __forceinline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                return *(this_t)base_t::Deserialize(Data);
            }
            static __forceinline this_t Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
                base_t::Deserialize(Data, Value);
            }

            __host__ __device__ __forceinline Field(const vector_t& Dimensions)
                : base_t(Dimensions) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline Field(_Ts... Dimensions)
                : Field(vector_t(Dimensions...)) { }
            __host__ __device__ __forceinline Field(const vector_t& Dimensions, const _T* All)
                : base_t(Dimensions) {
                CpyAllIn(All);
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline Field(_Ts... Dimensions, const _T* All)
                : Field(vector_t(Dimensions...), All) { }

            __host__ __device__ __forceinline Field(const this_t& Other)
                : Field(Other.Dimensions(), Other.Data()) { }
            __host__ __device__ __forceinline Field(this_t&& Other)
                : base_t(Other.Dimensions(), Other.Data()) {
                new (&Other) base_t(this->Dimensions(), 0);
            }
            __host__ __device__ __forceinline ~Field() {
                base_t::Dispose();
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
            __host__ __device__ __forceinline FieldProxyConst<_T, _DimensionCount> MakeProxy() const {
                return FieldProxyConst<_T, _DimensionCount>(*this);
            }

            __host__ __device__ __forceinline operator FieldProxy<_T, _DimensionCount>() {
                return MakeProxy();
            }
            __host__ __device__ __forceinline operator FieldProxyConst<_T, _DimensionCount>() const {
                return MakeProxy();
            }
        };
        template <typename _T, size_t _DimensionCount>
        class FieldProxy final : public details::FieldBase<_T, _DimensionCount> {
            using base_t = details::FieldBase<_T, _DimensionCount>;
            using vector_t = base_t::vector_t;
        public:
            using base_t::IdxToPtr;
            using base_t::IdxToRef;
            using base_t::CoordsToPtr;
            using base_t::CoordsToRef;
            using base_t::PtrToRef;
            using base_t::RefToPtr;
            using base_t::operator();
            using base_t::CpyAllIn;
            using base_t::CpyValIn;
            using base_t::CopyBlockIn;
            using base_t::Data;

            __host__ __device__ __forceinline Field<_T, _DimensionCount> Clone() const {
                return *(Field<_T, _DimensionCount>*)base_t::Clone();
            }

            __host__ __device__ FieldProxy(const vector_t& Dimensions, _T* All)
                : base_t(Dimensions, All) { }
            __host__ __device__ FieldProxy(Field<_T, _DimensionCount>& Parent)
                : base_t(Parent.Dimensions(), Parent.Data()) { }
            __host__ __device__ operator FieldProxyConst<_T, _DimensionCount>() const {
                return FieldProxyConst<_T, _DimensionCount>(*this);
            }
        };
        template <typename _T, size_t _DimensionCount>
        class FieldProxyConst final : public details::FieldBase<_T, _DimensionCount> {
            using base_t = details::FieldBase<_T, _DimensionCount>;
            using vector_t = base_t::vector_t;
        public:
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