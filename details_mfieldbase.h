#pragma once

#include "fields_field.h"

namespace bcuda {
    namespace details {
        template <template <typename, uintmax_t> typename _TFunction, uintmax_t _StartIndex, typename... _Ts>
        struct RunFunctionsOverTypeWrapper {
            template <typename... _TParameters>
            static void RunFunctionsOverType(_TParameters... Params) { }
        };
        template <template <typename, uintmax_t> typename _TFunction, uintmax_t _StartIndex, typename _T1, typename... _Ts>
        struct RunFunctionsOverTypeWrapper<_TFunction, _StartIndex, _T1, _Ts...> {
            template <typename... _TParameters>
            static void RunFunctionsOverType(_TParameters... Params) {
                _TFunction<_T1, _StartIndex>::Run(Params...);
                RunFunctionsOverTypeWrapper<_TFunction, _StartIndex + 1, _Ts...>::template RunFunctionsOverType<_TParameters...>(Params...);
            }
        };

        template <typename _T, uintmax_t _Idx>
        struct MFieldBase_MallocMem {
            __host__ __device__ static void Run(void** Dests, size_t VCount) {
#ifdef __CUDA_ARCH__
                Dests[_Idx] = (_T*)malloc(VCount * sizeof(_T));
#else
                ThrowIfBad(cudaMalloc(Dests + _Idx, VCount * sizeof(_T)));
#endif
            }
        };

        template <typename _T, uintmax_t _Idx>
        struct MFieldBase_Clone {
            __host__ __device__ static void Run(void** DestsD, void** DestsS, size_t VCount) {
#ifdef __CUDA_ARCH__
                memcpy(DestsD[_Idx], DestsS[_Idx], VCount * sizeof(_T));
#else
                ThrowIfBad(cudaMemcpy(DestsD[_Idx], DestsS[_Idx], VCount * sizeof(_T), cudaMemcpyDeviceToDevice));
#endif
            }
        };

        template <size_t _DimensionCount>
        struct MFieldBase_SerializedSize_Wrapper {
            template <typename _T, uintmax_t _Idx>
            struct MFieldBase_SerializedSize {
                static void Run(void** Arrs, size_t VCount, const FixedVector<uint32_t, _DimensionCount>& Dimensions, size_t& Total) {
                    Total += fields::FieldProxyConst<_T, _DimensionCount>(Dimensions, Arrs[_Idx]).SerializedSize();
                }
            };
        };

        template <size_t _DimensionCount>
        struct MFieldBase_Serialize_Wrapper {
            template <typename _T, uintmax_t _Idx>
            struct MFieldBase_Serialize {
                static void Run(void** Arrs, size_t VCount, const FixedVector<uint32_t, _DimensionCount>& Dimensions, void*& Data) {
                    fields::FieldProxyConst<_T, _DimensionCount>(Dimensions, Arrs[_Idx]).Serialize(Data);
                }
            };
        };

        template <size_t _DimensionCount>
        struct MFieldBase_Deserialize_Wrapper {
            template <typename _T, uintmax_t _Idx>
            struct MFieldBase_Deserialize {
                static void Run(void** Arrs, size_t VCount, const FixedVector<uint32_t, _DimensionCount>& Dimensions, const void*& Data) {
                    auto field = fields::FieldProxy<_T, _DimensionCount>(Dimensions, Arrs[_Idx]);
                    for (size_t i = 0; i < VCount; ++i)
                        field.CpyValIn(i, BSerializer::Deserialize<_T>(Data));
                }
            };
        };

        template <size_t _DimensionCount, typename... _Ts>
        class MFieldBase : public DimensionedBase<_DimensionCount> {
            static_assert(sizeof...(_Ts), "The parameter pack _Ts must contain at least one element.");

            using this_t = MFieldBase<_DimensionCount, _Ts...>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        public:
            using tuple_t = std::tuple<_Ts...>;
            template <size_t _Idx>
            using element_t = std::tuple_element_t<_Idx, tuple_t>;

            __host__ __device__ inline MFieldBase(const typename this_t::vector_t& Dimensions)
                : basedb_t(Dimensions) {
                if (!this->Length(0)) {
                    for (size_t i = 0; i < sizeof...(_Ts); ++i)
                        darrs[i] = 0;
                    return;
                }
                RunFunctionsOverTypeWrapper<MFieldBase_MallocMem, 0, _Ts...>::template RunFunctionsOverType((void**)&darrs, basedb_t::ValueCount());
            }
            __host__ __device__ inline MFieldBase(const typename this_t::vector_t& Dimensions, void* const* Arrays)
                : basedb_t(Dimensions) {
                for (size_t i = 0; i < sizeof...(_Ts); ++i)
                    darrs[i] = Arrays[i];
            }

            __host__ __device__ inline size_t EachValueCount() const {
                return basedb_t::ValueCount();
            }
            template <size_t _Idx>
            __host__ __device__ inline size_t EachSizeOnGPU() const {
                return EachValueCount() * sizeof(element_t<_Idx>);
            }
            __host__ __device__ inline size_t TotalValueCount() const {
                return EachValueCount() * sizeof...(_Ts);
            }
            __host__ __device__ inline size_t TotalSizeOnGPU() const {
                return EachValueCount() * (sizeof(_Ts) + ...);
            }

#pragma region ProxyAccess
            template <size_t _Idx>
            __host__ __device__ inline fields::FieldProxy<element_t<_Idx>, _DimensionCount> F() const {
                return fields::FieldProxy<element_t<_Idx>, _DimensionCount>(this->Dimensions(), (element_t<_Idx>*)darrs[_Idx]);
            }
            template <size_t _Idx>
            __host__ __device__ inline fields::FieldProxyConst<element_t<_Idx>, _DimensionCount> FConst() const {
                return fields::FieldProxyConst<element_t<_Idx>, _DimensionCount>(this->Dimensions(), (element_t<_Idx>*)darrs[_Idx]);
            }
            template <size_t _Idx>
            __host__ __device__ inline element_t<_Idx>* FData() const {
                return (element_t<_Idx>*)darrs[_Idx];
            }
#pragma endregion

            __host__ __device__ inline void Dispose() {
                for (size_t i = 0; i < sizeof...(_Ts); ++i) {
#ifdef __CUDA_ARCH__
                    free(darrs[i]);
#else
                    ThrowIfBad(cudaFree(darrs[i]));
#endif
                    darrs[i] = 0;
                }
            }

            __host__ __device__ inline this_t Clone() const {
                this_t clone(this->Dimensions());
                RunFunctionsOverTypeWrapper<MFieldBase_Clone, 0, _Ts...>::template RunFunctionsOverType(&clone.darrs, (void**)&darrs, basedb_t::ValueCount());
                return clone;
            }

            inline size_t SerializedSize() const requires (BSerializer::Serializable<_Ts> && ...) {
                size_t t = sizeof(typename this_t::vector_t);
                RunFunctionsOverTypeWrapper<MFieldBase_SerializedSize_Wrapper<_DimensionCount>::template MFieldBase_SerializedSize, 0, _Ts...>::template RunFunctionsOverType((void**)&darrs, basedb_t::ValueCount(), basedb_t::Dimensions(), t);
            }
            inline void Serialize(void*& Data) const requires (BSerializer::Serializable<_Ts> && ...) {
                BSerializer::Serialize(Data, basedb_t::Dimensions());
                RunFunctionsOverTypeWrapper<MFieldBase_Serialize_Wrapper<_DimensionCount>::template MFieldBase_Serialize, 0, _Ts...>::template RunFunctionOverType((void**)&darrs, basedb_t::ValueCount(), basedb_t::Dimensions(), Data);
            }
            static inline this_t Deserialize(const void*& Data) requires (BSerializer::Serializable<_Ts> && ...) {
                typename this_t::vector_t dims = BSerializer::Deserialize<typename this_t::vector_t>(Data);
                this_t value(dims);
                RunFunctionsOverTypeWrapper<MFieldBase_Deserialize_Wrapper<_DimensionCount>::template MFieldBase_Deserialize, 0, _Ts...>::template RunFunctionOverType((void**)&value.darrs, basedb_t::ValueCount(), basedb_t::Dimensions(), Data);
                return value;
            }
            static inline void Deserialize(const void*& Data, void* Value) requires (BSerializer::Serializable<_Ts> && ...) {
                new (Value) this_t(Deserialize(Data));
            }
            __host__ __device__ inline void* const* FieldDataArray() const {
                return darrs;
            }
        private:
            void* darrs[sizeof...(_Ts)];
        };
    }
}