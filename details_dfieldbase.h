#pragma once

#include "fields_field.h"

namespace BrendanCUDA {
    namespace details {
        template <typename _T, size_t _DimensionCount>
        class DFieldBase : public DimensionedBase<_DimensionCount> {
            using this_t = DFieldBase<_T, _DimensionCount>;
            using base_t = DimensionedBase<_DimensionCount>;
        public:
            __host__ __device__ __forceinline DFieldBase(vector_t Dimensions)
                : base_t(Dimensions) {
                if (!Length<0>()) {
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
            __host__ __device__ __forceinline DFieldBase(_Ts... Dimensions)
                : DFieldBase(vector_t(Dimensions...)) { }
            __host__ __device__ __forceinline DFieldBase(vector_t Dimensions, _T* ArrF, _T* ArrB)
                : base_t(Dimensions), darrF(ArrF), darrB(ArrB) { }
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DFieldBase(_Ts... Dimensions, _T* ArrF, _T* ArrB)
                : DFieldBase(vector_t(Dimensions...), ArrF, ArrB) { }

            using base_t::DimensionsD();
            __host__ __device__ __forceinline size_t SizeOnGPU() const {
                return ValueCount() * sizeof(_T);
            }

#pragma region ProxyAccess
        protected:
            __host__ __device__ Fields::FieldProxy<_T, _DimensionCount> F() {
                return Fields::FieldProxy<_T, _DimensionCount>(Dimensions(), darrF);
            }
            __host__ __device__ Fields::FieldProxy<_T, _DimensionCount> B() {
                return Fields::FieldProxy<_T, _DimensionCount>(Dimensions(), darrB);
            }
            __host__ __device__ _T* FData() {
                return darrF;
            }
            __host__ __device__ _T* BData() {
                return darrB;
            }
            __host__ __device__ void Reverse() {
                std::swap(darrF, darrB);
            }
        public:
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> F() const {
                return Fields::FieldProxyConst<_T, _DimensionCount>(Dimensions(), darrF);
            }
            __host__ __device__ Fields::FieldProxyConst<_T, _DimensionCount> B() const {
                return Fields::FieldProxyConst<_T, _DimensionCount>(Dimensions(), darrB);
            }
            __host__ __device__ const _T* FData() const {
                return darrF;
            }
            __host__ __device__ const _T* BData() const {
                return darrB;
            }
#pragma endregion

        protected:
            __host__ __device__ __forceinline void Dispose() {
#ifdef __CUDA_ARCH__
                free(darrF);
                free(darrB);
#else
                ThrowIfBad(cudaFree(darrF));
                ThrowIfBad(cudaFree(darrB));
#endif
            }
        public:

#pragma region CpyAll
        protected:
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyAllIn(const _T* All) {
                B().CpyAllIn<_CopyFromHost>(All);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllIn(const _T* All) {
                B().CpyAllIn(All);
            }
#endif
        public:
            template <bool _CopyToHost>
            __host__ __forceinline _T* CpyAllOut() const {
                return F().CpyAllOut<_CopyToHost>();
            }
#ifdef __CUDACC__
            __device__ __forceinline _T* CpyAllOut() const {
                return F().CpyAllOut();
            }
#endif
            template <bool _CopyToHost>
            __host__ __device__ __forceinline void CpyAllOut(_T* All) const {
                F().CpyAllOut<_CopyToHost>(All);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyAllOut(_T* All) const {
                F().CpyAllOut<_CopyToHost>(All);
            }
#endif
#pragma endregion

#pragma region CpyVal
        protected:
            __host__ __device__ __forceinline void CpyValIn(uint64_t Idx, const _T& Val) {
                B().CpyValIn(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T& Val) {
                B().CpyValIn(Coords, Val);
            }
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) {
                B().CpyValIn<_CopyFromHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(uint64_t Idx, const _T* Val) {
                B().CpyValIn(Idx, Val);
            }
#endif
            template <bool _CopyFromHost>
            __host__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) {
                B().CpyValIn<_CopyFromHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValIn(const vector_t& Coords, const _T* Val) {
                B().CpyValIn(Coords, Val);
            }
#endif
        public:
            __host__ __device__ __forceinline void CpyValOut(uint64_t Idx, _T& Val) const {
                F().CpyValOut(Idx, Val);
            }
            __host__ __device__ __forceinline void CpyValOut(const vector_t& Coords, _T& Val) const {
                F().CpyValOut(Coords, Val);
            }
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                F().CpyValOut<_CopyToHost>(Idx, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(uint64_t Idx, _T* Val) const {
                F().CpyValOut(Idx, Val);
            }
#endif
            template <bool _CopyToHost>
            __host__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                F().CpyValOut<_CopyToHost>(Coords, Val);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CpyValOut(const vector_t& Coords, _T* Val) const {
                F().CpyValOut(Coords, Val);
            }
#endif
            __host__ __device__ __forceinline _T CpyValOut(uint64_t Idx) const {
                F().CpyValOut(Idx);
            }
            __host__ __device__ __forceinline _T CpyValOut(const vector_t& Coords) const {
                F().CpyValOut(Coords);
            }
#pragma endregion

        protected:
            template <bool _InputOnHost>
            __host__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) {
                B().CopyBlockIn<_InputOnHost>(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockIn(const _T* Input, const vector_t& InputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) {
                B().CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif
        public:
            template <bool _OutputOnHost>
            __host__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                F().CopyBlockOut<_OutputOnHost>(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#ifdef __CUDACC__
            __device__ __forceinline void CopyBlockOut(_T* Output, const vector_t& OutputDimensions, const vector_t& RangeDimensions, const vector_t& RangeInInputsCoordinates, const vector_t& RangeInOutputsCoordinates) const {
                F().CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
            }
#endif

            __host__ __device__ this_t Clone() const {
                this_t clone(Dimensions());
                clone.F().CpyAllIn<false>(FData());
                return clone;
            }

            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T> {
                F().SerializedSize();
            }
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
                F().Serialize(Data);
            }
        protected:
            static __forceinline this_t Deserialize(const void*& Data) requires BSerializer::Serializable<_T> {
                FieldBase<_T, _DimensionCount> field = FieldBase<_T, _DimensionCount>::Deserialize(Data);
                this_t value(field.Dimensions());
                value.F().CpyAllIn<false>(field.Data());
                return value;
            }
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
                new (Value) this_t(Deserialize(Data));
            }
        public:
        private:
            _T* darrF;
            _T* darrB;
        };
    }
}