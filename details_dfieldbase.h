#pragma once

#include "fields_field.h"

namespace bcuda {
    namespace details {
        template <typename _T, size_t _DimensionCount>
        class DFieldBase : public DimensionedBase<_DimensionCount> {
            using this_t = DFieldBase<_T, _DimensionCount>;
            using basedb_t = DimensionedBase<_DimensionCount>;
        public:
            __host__ __device__ inline DFieldBase(const typename this_t::vector_t& Dimensions)
                : basedb_t(Dimensions) {
                if (!this->Length(0)) {
                    darrF = 0;
                    darrB = 0;
                    return;
                }
#ifdef __CUDA_ARCH__
                darrF = (_T*)malloc(SizeOnGPU());
                darrB = (_T*)malloc(SizeOnGPU());
#else
                ThrowIfBad(cudaMalloc(&darrF, SizeOnGPU()));
                ThrowIfBad(cudaMalloc(&darrB, SizeOnGPU()));
#endif
            }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ inline DFieldBase(_Ts... Dimensions)
                : DFieldBase(typename this_t::vector_t(Dimensions...)) { }
            __host__ __device__ inline DFieldBase(const typename this_t::vector_t& Dimensions, _T* ArrF, _T* ArrB)
                : basedb_t(Dimensions), darrF(ArrF), darrB(ArrB) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ inline DFieldBase(_Ts... Dimensions, _T* ArrF, _T* ArrB)
                : DFieldBase(typename this_t::vector_t(Dimensions...), ArrF, ArrB) { }

            __host__ __device__ inline size_t SizeOnGPU() const {
                return basedb_t::ValueCount() * sizeof(_T);
            }

#pragma region ProxyAccess
            __host__ __device__ fields::FieldProxy<_T, _DimensionCount> F() const {
                return fields::FieldProxy<_T, _DimensionCount>(this->Dimensions(), darrF);
            }
            __host__ __device__ inline fields::FieldProxy<_T, _DimensionCount> B() const {
                return fields::FieldProxy<_T, _DimensionCount>(this->Dimensions(), darrB);
            }
            __host__ __device__ inline fields::FieldProxyConst<_T, _DimensionCount> FConst() const {
                return fields::FieldProxyConst<_T, _DimensionCount>(this->Dimensions(), darrF);
            }
            __host__ __device__ inline fields::FieldProxyConst<_T, _DimensionCount> BConst() const {
                return fields::FieldProxyConst<_T, _DimensionCount>(this->Dimensions(), darrB);
            }
            __host__ __device__ inline _T* FData() const {
                return darrF;
            }
            __host__ __device__ inline _T* BData() const {
                return darrB;
            }
            __host__ __device__ inline void Reverse() {
                std::swap(darrF, darrB);
            }
#pragma endregion

            __host__ __device__ inline void Dispose() {
#ifdef __CUDA_ARCH__
                free(darrF);
                free(darrB);
#else
                ThrowIfBad(cudaFree(darrF));
                ThrowIfBad(cudaFree(darrB));
#endif
            }

#pragma region CpyAll
            template <bool _CopyFromHost>
            __host__ inline void CpyAllIn(const _T* All) const {
                B().CpyAllIn<_CopyFromHost>(All);
            }
#ifdef __CUDACC__
            __device__ inline void CpyAllIn(const _T* All) const {
                B().CpyAllIn(All);
            }
#endif
            template <bool _CopyToHost>
            __host__ inline _T* CpyAllOut() const {
                return F().CpyAllOut<_CopyToHost>();
            }
#ifdef __CUDACC__
            __device__ inline _T* CpyAllOut() const {
                return F().CpyAllOut();
            }
#endif
            template <bool _CopyToHost>
            __host__ __device__ inline void CpyAllOut(_T* All) const {
                F().CpyAllOut<_CopyToHost>(All);
            }
#ifdef __CUDACC__
            __device__ inline void CpyAllOut(_T* All) const {
                F().CpyAllOut(All);
            }
#endif
#pragma endregion

#pragma region CpyVal
            __host__ __device__ inline void CpyValIn(uint64_t Idx, const _T& Val) const {
                B().CpyValIn(Idx, Val);
            }
            __host__ __device__ inline void CpyValIn(const typename this_t::vector_t& Coords, const _T& Val) const {
                B().CpyValIn(Coords, Val);
            }
            template <bool _CopyFromHost>
            __host__ inline void CpyValIn(uint64_t Idx, const _T* Val) const {
                B().CpyValIn<_CopyFromHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ inline void CpyValIn(uint64_t Idx, const _T* Val) {
                B().CpyValIn(Idx, Val);
            }
#endif
            template <bool _CopyFromHost>
            __host__ inline void CpyValIn(const typename this_t::vector_t& Coords, const _T* Val) const {
                B().CpyValIn<_CopyFromHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ inline void CpyValIn(const typename this_t::vector_t& Coords, const _T* Val) const {
                B().CpyValIn(Coords, Val);
            }
#endif
            __host__ __device__ inline void CpyValOut(uint64_t Idx, _T& Val) const {
                F().CpyValOut(Idx, Val);
            }
            __host__ __device__ inline void CpyValOut(const typename this_t::vector_t& Coords, _T& Val) const {
                F().CpyValOut(Coords, Val);
            }
            template <bool _CopyToHost>
            __host__ inline void CpyValOut(uint64_t Idx, _T* Val) const {
                F().CpyValOut<_CopyToHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ inline void CpyValOut(uint64_t Idx, _T* Val) const {
                F().CpyValOut(Idx, Val);
            }
#endif
            template <bool _CopyToHost>
            __host__ inline void CpyValOut(const typename this_t::vector_t& Coords, _T* Val) const {
                F().CpyValOut<_CopyToHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ inline void CpyValOut(const typename this_t::vector_t& Coords, _T* Val) const {
                F().CpyValOut(Coords, Val);
            }
#endif
            __host__ __device__ inline _T CpyValOut(uint64_t Idx) const {
                F().CpyValOut(Idx);
            }
            __host__ __device__ inline _T CpyValOut(const typename this_t::vector_t& Coords) const {
                F().CpyValOut(Coords);
            }
#pragma endregion

            template <bool _InputOnHost>
            __host__ inline void CopyBlockIn(const _T* Input, const typename this_t::vector_t& InputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                B().CopyBlockIn<_InputOnHost>(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ inline void CopyBlockIn(const _T* Input, const typename this_t::vector_t& InputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                B().CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
            template <bool _OutputOnHost>
            __host__ inline void CopyBlockOut(_T* Output, const typename this_t::vector_t& OutputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                F().CopyBlockOut<_OutputOnHost>(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ inline void CopyBlockOut(_T* Output, const typename this_t::vector_t& OutputDimensions, const typename this_t::vector_t& RangeDimensions, const typename this_t::vector_t& RangeInInputsCoordinates, const typename this_t::vector_t& RangeInOutputsCoordinates) const {
                F().CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif

            __host__ __device__ inline this_t Clone() const {
                this_t clone(this->Dimensions());
                clone.F().CpyAllIn<false>(FData());
                return clone;
            }

            inline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                F().SerializedSize();
            }
            inline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                F().Serialize(Data);
            }
            static inline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                FieldBase<_T, _DimensionCount> field = FieldBase<_T, _DimensionCount>::Deserialize(Data);
                this_t value(field.Dimensions());
                value.F().CpyAllIn<false>(field.Data());
                return value;
            }
            static inline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
                new (Value) this_t(Deserialize(Data));
            }
        private:
            _T* darrF;
            _T* darrB;
        };
    }
}