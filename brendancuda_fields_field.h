#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/device_ptr.h>

#include "brendancuda_dcopy.cuh"
#include "brendancuda_points.h"
#include "brendancuda_errorhelp.h"
#include "brendancuda_cudaconstexpr.h"
#include "brendancuda_copyblock.h"
#include "brendancuda_details_fillwith.h"
#include "BSerializer/Serializer.h"

namespace BrendanCUDA {
    namespace Fields {
        template <typename _T, size_t _DimensionCount>
        class Field final {
            static_assert(_DimensionCount, "_DimensionCount may not be zero.");
            using vector_t = FixedVector<uint32_t, _DimensionCount>;
        public:
            __host__ __device__ __forceinline Field(vector_t Dimensions);
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline Field(_Ts... Dimensions);
            __host__ __device__ __forceinline Field(vector_t Dimensions, _T* All);
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline Field(_Ts... Dimensions, _T* All);

            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4);
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4);
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4);
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4);

            template <size_t _Index>
                requires (_Index < _DimensionCount)
            __host__ __device__ __forceinline uint32_t Length() const;

            __host__ __device__ __forceinline vector_t Dimensions() const;

            __host__ __device__ __forceinline size_t ValueCount() const;
            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(vector_t Coordinates);
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(_Ts... Coordinates);

            __host__ __device__ __forceinline void CopyAllIn(_T* All);
            __host__ __device__ __forceinline void CopyAllOut(_T* All) const;
            __host__ __device__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
            __host__ __device__ __forceinline void CopyValueIn(vector_t Coordinates, _T* Value);
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline void CopyValueIn(_Ts... Coordinates, _T* Value);
            __host__ __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ __device__ __forceinline void CopyValueOut(vector_t Coordinates, _T* Value) const;
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline void CopyValueOut(_Ts... Coordinates, _T* Value) const;

            template <bool _CopyToHost>
            __host__ __forceinline _T* GetAll() const;
#ifdef __CUDACC__
            __device__ __forceinline _T* GetAll() const;
#endif
            __host__ __device__ __forceinline void SetAll(_T* All);

            __host__ __device__ __forceinline _T GetValueAt(uint64_t Index) const;
            __host__ __device__ __forceinline _T GetValueAt(vector_t Coordinates) const;
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline _T GetValueAt(_Ts... Coordinates) const;

            __host__ __device__ __forceinline void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ __forceinline void SetValueAt(vector_t Coordinates, _T Value);
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline void SetValueAt(_Ts... Coordinates, _T Value);

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(vector_t Coordinates) const;
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(_Ts... Coordinates) const;
            __host__ __device__ __forceinline vector_t IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ __forceinline _T* IndexToPointer(uint64_t Index) const;
            __host__ __device__ __forceinline uint64_t PointerToIndex(_T* Pointer) const;

            __host__ __device__ __forceinline _T* CoordinatesToPointer(vector_t Coordinates) const;
            template <std::same_as<_T>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline _T* CoordinatesToPointer(_Ts... Coordinates) const;
            __host__ __device__ __forceinline vector_t PointerToCoordinates(_T* Pointer) const;

            __host__ __device__ __forceinline void FillWith(_T Value);

            __host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> Data() const;

            __host__ __device__ __forceinline void CopyBlockIn(_T* Input, vector_t InputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates);
            __host__ __device__ __forceinline void CopyBlockOut(_T* Output, vector_t OutputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates);

            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T>;
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T>;
            static __forceinline Field<_T, _DimensionCount> Deserialize(const void*& Data) requires BSerializer::Serializable<_T>;
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T>;
        private:
            vector_t dimensions;
            _T* cudaArray;
        };
    }
}

template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field<_T, _DimensionCount>::Field(vector_t Dimensions) {
    for (size_t i = 0; i < _DimensionCount; ++i)
        if (!Dimensions[i]) {
            dimensions = vector_t();
            cudaArray = 0;
            return;
        }
    dimensions = Dimensions;
#ifdef __CUDA_ARCH__
    cudaArray = malloc(SizeOnGPU());
#else
    ThrowIfBad(cudaMalloc(&cudaArray, SizeOnGPU()));
#endif
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline BrendanCUDA::Fields::Field<_T, _DimensionCount>::Field(_Ts... Dimensions)
    : Field(vector_t(Dimensions...)) { }
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field<_T, _DimensionCount>::Field(vector_t Dimensions, _T* All)
    : Field(Dimensions) {
    CopyAllIn(All);
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline BrendanCUDA::Fields::Field<_T, _DimensionCount>::Field(_Ts... Dimensions, _T* All)
    : Field(vector_t(Dimensions...), All) { }
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::LengthX() const requires (_DimensionCount <= 4) {
    return dimensions.x;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
    return dimensions.y;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
    return dimensions.z;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::LengthW() const requires (_DimensionCount == 4) {
    return dimensions.w;
}
template <typename _T, size_t _DimensionCount>
template <size_t _Index>
    requires (_Index < _DimensionCount)
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::Length() const {
    return dimensions[_Index];
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::Fields::Field<_T, _DimensionCount>::Dimensions() const -> vector_t {
    return dimensions;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::ValueCount() const {
    size_t s = 1;
    for (size_t i = 0; i < _DimensionCount; ++i)
        s *= dimensions[i];
    return s;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::SizeOnGPU() const {
    return sizeof(_T) * ValueCount();
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::Fields::Field<_T, _DimensionCount>::operator()(vector_t Coordinates) {
    uint64_t idx = BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Coordinates);
#ifdef __CUDA_ARCH__
    return cudaArray[idx];
#else
    return *thrust::device_ptr<_T>(cudaArray + idx);
#endif
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::Fields::Field<_T, _DimensionCount>::operator()(_Ts... Coordinates) {
    uint64_t idx = BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), vector_t(Coordinates...));
#ifdef __CUDA_ARCH__
    return cudaArray[idx];
#else
    return *thrust::device_ptr<_T>(cudaArray + idx);
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyAllIn(_T* All) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(cudaArray, All, SizeOnGPU());
#else
    ThrowIfBad(cudaMemcpy(cudaArray, All, SizeOnGPU(), cudaMemcpyDefault));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyAllOut(_T* All) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(All, cudaArray, SizeOnGPU());
#else
    ThrowIfBad(cudaMemcpy(All, cudaArray, SizeOnGPU(), cudaMemcpyDefault));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyValueIn(uint64_t Index, _T* Value) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(IndexToPointer(Index), Value, sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyValueIn(vector_t Coordinates, _T* Value) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyValueIn(_Ts... Coordinates, _T* Value) {
    CopyValueIn(vector_t(Coordinates...), Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyValueOut(uint64_t Index, _T* Value) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(Value, IndexToPointer(Index), sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyValueOut(vector_t Coordinates, _T* Value) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyValueOut(_Ts... Coordinates, _T* Value) const {
    CopyValueOut(vector_t(Coordinates...), Value);
}
template <typename _T, size_t _DimensionCount>
template <bool _CopyToHost>
__host__ __forceinline _T* BrendanCUDA::Fields::Field<_T, _DimensionCount>::GetAll() const {
    _T* a;
    if constexpr (_CopyToHost) {
        a = (_T*)malloc(SizeOnGPU());
    }
    else {
        ThrowIfBad(cudaMalloc(&a, SizeOnGPU()));
    }
    CopyAllOut(a);
    return a;
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline _T* BrendanCUDA::Fields::Field<_T, _DimensionCount>::GetAll() const {
    _T* a = (_T*)malloc(SizeOnGPU());
    CopyAllOut(a, false);
    return a;
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::SetAll(_T* All) {
    CopyAllIn(All);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field<_T, _DimensionCount>::GetValueAt(uint64_t Index) const {
    _T v;
    CopyValueOut(Index, &v);
    return v;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field<_T, _DimensionCount>::GetValueAt(vector_t Coordinates) const {
    _T v;
#ifdef __CUDA_ARCH__
    CopyValueOut(Coordinates, &v);
#else
    CopyValueOut(Coordinates, &v, true);
#endif
    return v;
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field<_T, _DimensionCount>::GetValueAt(_Ts... Coordinates) const {
    return GetValueAt(vector_t(Coordinates...));
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::SetValueAt(uint64_t Index, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(Index, &Value);
#else
    CopyValueIn(Index, &Value, true);
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::SetValueAt(vector_t Coordinates, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(Coordinates, &Value);
#else
    CopyValueIn(Coordinates, &Value, true);
#endif
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::SetValueAt(_Ts... Coordinates, _T Value) {
    SetValueAt(vector_t(Coordinates...), Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::Dispose() {
#ifdef __CUDA_ARCH__
    free(cudaArray);
#else
    ThrowIfBad(cudaFree(cudaArray));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::CoordinatesToIndex(vector_t Coordinates) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Coordinates);
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::CoordinatesToIndex(_Ts... Coordinates) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), vector_t(Coordinates...));
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::Fields::Field<_T, _DimensionCount>::IndexToCoordinates(uint64_t Index) const -> vector_t {
    return BrendanCUDA::IndexToCoordinates<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Index);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field<_T, _DimensionCount>::IndexToPointer(uint64_t Index) const {
    return &cudaArray[Index];
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::PointerToIndex(_T* Pointer) const {
    return Pointer - cudaArray;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field<_T, _DimensionCount>::CoordinatesToPointer(vector_t Coordinates) const {
    return IndexToPointer(CoordinatesToIndex(Coordinates));
}
template <typename _T, size_t _DimensionCount>
template <std::same_as<_T>... _Ts>
    requires (sizeof...(_Ts) == _DimensionCount)
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field<_T, _DimensionCount>::CoordinatesToPointer(_Ts... Coordinates) const {
    return IndexToPointer(CoordinatesToIndex(vector_t(Coordinates...)));
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::Fields::Field<_T, _DimensionCount>::PointerToCoordinates(_T* Pointer) const -> vector_t {
    return IndexToCoordinates(PointerToIndex(Pointer));
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::FillWith(_T Value) {
    BrendanCUDA::FillWith(cudaArray, ValueCount(), Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> BrendanCUDA::Fields::Field<_T, _DimensionCount>::Data() const {
    return { thrust::device_ptr<_T>(cudaArray), ValueCount() };
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyBlockIn(_T* Input, vector_t InputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates) {
    CopyBlock<_T, _DimensionCount, true>(Input, cudaArray, InputDimensions, Dimensions(), RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::CopyBlockOut(_T* Output, vector_t OutputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates) {
    CopyBlock<_T, _DimensionCount, true>(cudaArray, Output, Dimensions(), OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T, size_t _DimensionCount>
__forceinline size_t BrendanCUDA::Fields::Field<_T, _DimensionCount>::SerializedSize() const requires BSerializer::Serializable<_T> {
    size_t t = sizeof(uint32_t) * _DimensionCount;
    size_t l = ValueCount();
    for (size_t i = 0; i < l; ++i)
        t += BSerializer::SerializedSize(GetValueAt(i));
    return t;
}
template <typename _T, size_t _DimensionCount>
__forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
    BSerializer::Serialize(dimensions);
    size_t l = ValueCount();
    for (size_t i = 0; i < l; ++i)
        BSerializer::Serialize(Data, GetValueAt(i));
}
template <typename _T, size_t _DimensionCount>
__forceinline auto BrendanCUDA::Fields::Field<_T, _DimensionCount>::Deserialize(const void*& Data) -> Field<_T, _DimensionCount> requires BSerializer::Serializable<_T> {
    vector_t dimensions = BSerializer::Deserialize<vector_t>(Data);
    Field<_T, _DimensionCount> field(dimensions);
    size_t l = field.ValueCount();
    for (size_t i = 0; i < l; ++i)
        field.SetValueAt(i, BSerializer::Deserialize<_T>(Data));
    return field;
}
template <typename _T, size_t _DimensionCount>
__forceinline void BrendanCUDA::Fields::Field<_T, _DimensionCount>::Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
    new (Value) Field<_T, _DimensionCount>(Deserialize<Field<_T, _DimensionCount>>(Data));
}