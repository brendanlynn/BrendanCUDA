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
        template <typename _T>
        class Field2 final {
        public:
            __host__ __device__ __forceinline Field2(uint32_2 Dimensions);
            __host__ __device__ __forceinline Field2(uint32_t LengthX, uint32_t LengthY);

            __host__ __device__ __forceinline Field2(uint32_2 Dimensions, _T* All);
            __host__ __device__ __forceinline Field2(uint32_t LengthX, uint32_t LengthY, _T* All);

            __host__ __device__ __forceinline uint32_t LengthX() const;
            __host__ __device__ __forceinline uint32_t LengthY() const;

            __host__ __device__ __forceinline uint32_2 Dimensions() const;

            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(uint32_t X, uint32_t Y);

            __host__ __device__ __forceinline void CopyAllIn(_T* All);
            __host__ __device__ __forceinline void CopyAllOut(_T* All) const;
            __host__ __device__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
            __host__ __device__ __forceinline void CopyValueIn(uint32_2 Coordinates, _T* Value);
            __host__ __device__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, _T* Value);
            __host__ __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ __device__ __forceinline void CopyValueOut(uint32_2 Coordinates, _T* Value) const;
            __host__ __device__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const;

            template <bool _CopyToHost>
            __host__ __forceinline _T* GetAll() const;
#ifdef __CUDACC__
            __device__ __forceinline _T* GetAll() const;
#endif
            __host__ __device__ __forceinline void SetAll(_T* All);

            __host__ __device__ __forceinline _T GetValueAt(uint64_t Index) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_2 Coordinates) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_t X, uint32_t Y) const;

            __host__ __device__ __forceinline void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_2 Coordinates, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_t X, uint32_t Y, _T Value);

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_2 Coordinates) const;
            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_t X, uint32_t Y) const;
            __host__ __device__ __forceinline uint32_2 IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ __forceinline _T* IndexToPointer(uint64_t Index) const;
            __host__ __device__ __forceinline uint64_t PointerToIndex(_T* Pointer) const;

            __host__ __device__ __forceinline _T* CoordinatesToPointer(uint32_2 Coordinates) const;
            __host__ __device__ __forceinline _T* CoordinatesToPointer(uint32_t X, uint32_t Y) const;
            __host__ __device__ __forceinline uint32_2 PointerToCoordinates(_T* Pointer) const;

            __host__ __device__ __forceinline void FillWith(_T Value);

            __host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> Data() const;

            __host__ __device__ __forceinline void CopyBlockIn(_T* Input, uint32_2 InputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates);
            __host__ __device__ __forceinline void CopyBlockOut(_T* Output, uint32_2 OutputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates);

            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T>;
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T>;
            static __forceinline Field2<_T> Deserialize(const void*& Data) requires BSerializer::Serializable<_T>;
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T>;
        private:
            uint32_t lengthX;
            uint32_t lengthY;
            uint32_t lengthZ;

            _T* cudaArray;
        };
    }
}

template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_2 Dimensions)
    : Field2(Dimensions.x, Dimensions.y) { }
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_t LengthX, uint32_t LengthY) {
    if (LengthX == 0 || LengthY == 0) {
        lengthX = 0;
        lengthY = 0;
        cudaArray = 0;
    }
    else {
        lengthX = LengthX;
        lengthY = LengthY;
#ifdef __CUDA_ARCH__
        cudaArray = new _T[LengthX * LengthY];
#else
        ThrowIfBad(cudaMalloc(&cudaArray, SizeOnGPU()));
#endif
    }
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_2 Dimensions, _T* All)
    : Field2(Dimensions.x, Dimensions.y, All) { }
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_t LengthX, uint32_t LengthY, _T* All)
    : Field2(LengthX, LengthY) {
    CopyAllIn(All);
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field2<_T>::LengthX() const {
    return lengthX;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field2<_T>::LengthY() const {
    return lengthY;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_2 BrendanCUDA::Fields::Field2<_T>::Dimensions() const {
    return uint32_2(lengthX, lengthY);
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::Field2<_T>::SizeOnGPU() const {
    return (((size_t)lengthX) * ((size_t)lengthY)) * sizeof(_T);
}
template <typename _T>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::Fields::Field2<_T>::operator()(uint32_t X, uint32_t Y) {
    uint64_t idx = BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, 2, true>(Dimensions(), uint32_2(X, Y));
#ifdef __CUDA_ARCH__
    return cudaArray[idx];
#else
    return *thrust::device_ptr<_T>(cudaArray + idx);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyAllIn(_T* All) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(cudaArray, All, SizeOnGPU());
#else
    ThrowIfBad(cudaMemcpy(cudaArray, All, SizeOnGPU(), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyAllOut(_T* All) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(All, cudaArray, SizeOnGPU());
#else
    ThrowIfBad(cudaMemcpy(All, cudaArray, SizeOnGPU(), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint64_t Index, _T* Value) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(IndexToPointer(Index), Value, sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint32_2 Coordinates, _T* Value) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint32_t X, uint32_t Y, _T* Value) {
#ifdef __CUDA_ARCH__
    deviceMemcpy(CoordinatesToPointer(X, Y), Value, sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(X, Y), Value, sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(Value, IndexToPointer(Index), sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint32_2 Coordinates, _T* Value) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const {
#ifdef __CUDA_ARCH__
    deviceMemcpy(Value, CoordinatesToPointer(X, Y), sizeof(_T));
#else
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(X, Y), sizeof(_T), cudaMemcpyDefault));
#endif
}
template <typename _T>
template <bool _CopyToHost>
__host__ __forceinline _T* BrendanCUDA::Fields::Field2<_T>::GetAll() const {
    _T* a;
    if constexpr (_CopyToHost) {
        a = new _T[lengthX * lengthY];
    }
    else {
        ThrowIfBad(cudaMalloc(&a, SizeOnGPU()));
    }
    CopyAllOut(a);
    return a;
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::Fields::Field2<_T>::GetAll() const {
    _T* a = new _T[lengthX * lengthY];
    CopyAllOut(a, false);
    return a;
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::SetAll(_T* All) {
    CopyAllIn(All);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field2<_T>::GetValueAt(uint64_t Index) const {
    _T v;
    CopyValueOut(Index, &v);
    return v;
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field2<_T>::GetValueAt(uint32_2 Coordinates) const {
    return GetValueAt(Coordinates.x, Coordinates.y);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field2<_T>::GetValueAt(uint32_t X, uint32_t Y) const {
    _T v;
#ifdef __CUDA_ARCH__
    CopyValueOut(X, Y, &v);
#else
    CopyValueOut(X, Y, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::SetValueAt(uint64_t Index, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(Index, &Value);
#else
    CopyValueIn(Index, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::SetValueAt(uint32_2 Coordinates, _T Value) {
    SetValueAt(Coordinates.x, Coordinates.y, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::SetValueAt(uint32_t X, uint32_t Y, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(X, Y, &Value);
#else
    CopyValueIn(X, Y, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::Dispose() {
#ifdef __CUDA_ARCH__
    delete[] cudaArray;
#else
    ThrowIfBad(cudaFree(cudaArray));
#endif
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field2<_T>::CoordinatesToIndex(uint32_2 Coordinates) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, 2, true>(Dimensions(), Coordinates);
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field2<_T>::CoordinatesToIndex(uint32_t X, uint32_t Y) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, 2, true>(Dimensions(), uint32_2(X, Y));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_2 BrendanCUDA::Fields::Field2<_T>::IndexToCoordinates(uint64_t Index) const {
    return BrendanCUDA::IndexToCoordinates<uint64_t, uint32_t, 2, true>(Dimensions(), Index);
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field2<_T>::IndexToPointer(uint64_t Index) const {
    return &cudaArray[Index];
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field2<_T>::PointerToIndex(_T* Pointer) const {
    return Pointer - cudaArray;
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field2<_T>::CoordinatesToPointer(uint32_2 Coordinates) const {
    return IndexToPointer(CoordinatesToIndex(Coordinates));
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field2<_T>::CoordinatesToPointer(uint32_t X, uint32_t Y) const {
    return IndexToPointer(CoordinatesToIndex(X, Y));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_2 BrendanCUDA::Fields::Field2<_T>::PointerToCoordinates(_T* Pointer) const {
    return IndexToCoordinates(PointerToIndex(Pointer));
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::FillWith(_T Value) {
    BrendanCUDA::FillWith(cudaArray, lengthX * lengthY, Value);
}
template <typename _T>
__host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> BrendanCUDA::Fields::Field2<_T>::Data() const {
    return { thrust::device_ptr<_T>(cudaArray), (size_t)lengthX * (size_t)lengthY };
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyBlockIn(_T* Input, uint32_2 InputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates) {
    CopyBlock<_T, 2, true>(Input, cudaArray, InputDimensions, Dimensions(), RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyBlockOut(_T* Output, uint32_2 OutputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates) {
    CopyBlock<_T, 2, true>(cudaArray, Output, Dimensions(), OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T>
__forceinline size_t BrendanCUDA::Fields::Field2<_T>::SerializedSize() const requires BSerializer::Serializable<_T> {
    size_t t = sizeof(uint32_t) << 1;
    size_t l = lengthX * lengthY;
    for (size_t i = 0; i < l; ++i)
        t += BSerializer::SerializedSize(GetValueAt(i));
    return t;
}
template <typename _T>
__forceinline void BrendanCUDA::Fields::Field2<_T>::Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
    BSerializer::Serialize(lengthX);
    BSerializer::Serialize(lengthY);
    size_t l = lengthX * lengthY;
    for (size_t i = 0; i < l; ++i)
        BSerializer::Serialize(Data, GetValueAt(i));
}
template <typename _T>
__forceinline auto BrendanCUDA::Fields::Field2<_T>::Deserialize(const void*& Data) -> Field2<_T> requires BSerializer::Serializable<_T> {
    uint32_t lengthX = BSerializer::Deserialize<uint32_t>(Data);
    uint32_t lengthY = BSerializer::Deserialize<uint32_t>(Data);
    Field2<_T> field(lengthX, lengthY);
    size_t l = lengthX * lengthY;
    for (size_t i = 0; i < l; ++i)
        field.SetValueAt(i, BSerializer::Deserialize<_T>(Data));
    return field;
}
template <typename _T>
__forceinline void BrendanCUDA::Fields::Field2<_T>::Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
    new (Value) Field2<_T>(Deserialize(Data));
}