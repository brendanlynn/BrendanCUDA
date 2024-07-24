#pragma once

#include "brendancuda_fields_field2.h"
#include "brendancuda_points.h"
#include "brendancuda_errorhelp.h"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <typename _T>
        class DField2 final {
        public:
            __host__ __device__ __forceinline DField2(uint32_2 Dimensions);
            __host__ __device__ __forceinline DField2(uint32_t LengthX, uint32_t LengthY);

            __host__ __device__ __forceinline DField2(uint32_2 Dimensions, _T* All);
            __host__ __device__ __forceinline DField2(uint32_t LengthX, uint32_t LengthY, _T* All);

            __host__ __device__ __forceinline uint32_t LengthX() const;
            __host__ __device__ __forceinline uint32_t LengthY() const;

            __host__ __device__ __forceinline uint32_2 Dimensions() const;

            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline Field2<_T> FFront() const;
            __host__ __device__ __forceinline Field2<_T> FBack() const;
            __host__ __device__ __forceinline void Reverse();

            __host__ __forceinline void CopyAllIn(_T* All);
            __host__ __forceinline void CopyAllOut(_T* All) const;
            __host__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
            __host__ __forceinline void CopyValueIn(uint32_2 Coordinates, _T* Value);
            __host__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, _T* Value);
            __host__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ __forceinline void CopyValueOut(uint32_2 Coordinates, _T* Value) const;
            __host__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const;

            __host__ __forceinline _T* GetAll(bool CopyToHost) const;
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

            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_2 Coordinates) const;
            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_t X, uint32_t Y) const;
            __host__ __device__ __forceinline uint32_2 IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ __forceinline void FillWith(_T Value);

            __host__ __device__ __forceinline void CopyBlockIn(_T* Input, uint32_2 InputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates);
            __host__ __device__ __forceinline void CopyBlockOut(_T* Output, uint32_2 OutputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates);
        private:
            uint32_t lengthX;
            uint32_t lengthY;

            _T* cudaArrayF;
            _T* cudaArrayB;
        };
    }
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_2 Dimensions)
    : DField2(Dimensions.x, Dimensions.y) { }
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_t LengthX, uint32_t LengthY) {
    if (LengthX == 0 || LengthY == 0) {
        lengthX = 0;
        lengthY = 0;
        cudaArrayF = 0;
        cudaArrayB = 0;
    }
    else {
        lengthX = LengthX;
        lengthY = LengthY;
#ifdef __CUDA_ARCH__
        size_t l = (size_t)LengthX * (size_t)LengthY;
        cudaArrayF = new _T[l];
        cudaArrayB = new _T[l];
#else
        size_t l = (size_t)LengthX * (size_t)LengthY * sizeof(_T);
        ThrowIfBad(cudaMalloc(&cudaArrayF, l));
        ThrowIfBad(cudaMalloc(&cudaArrayB, l));
#endif
    }
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_2 Dimensions, _T* All)
    : DField2(Dimensions.x, Dimensions.y, All) { }
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_t LengthX, uint32_t LengthY, _T* All)
    : DField2(LengthX, LengthY) {
    CopyAllIn(All);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::Dispose() {
#ifdef __CUDA_ARCH__
    delete[] cudaArrayF;
    delete[] cudaArrayB;
#else
    ThrowIfBad(cudaFree(cudaArrayF));
    ThrowIfBad(cudaFree(cudaArrayB));
#endif
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field2<_T> BrendanCUDA::Fields::DField2<_T>::FFront() const {
    return *(Field2<_T>*)this;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field2<_T> BrendanCUDA::Fields::DField2<_T>::FBack() const {
    uint8_t r[sizeof(Field2<_T>)];
    *(uint64_t*)r = *(uint64_t*)this;
    ((uint32_t*)r)[2] = ((uint32_t*)this)[2];
    ((void**)r)[2] = ((void**)this)[3];
    return *(Field2<_T>*) & r;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::Reverse() {
    _T* i = cudaArrayF;
    cudaArrayF = cudaArrayB;
    cudaArrayB = i;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField2<_T>::LengthX() const {
    return lengthX;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField2<_T>::LengthY() const {
    return lengthY;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_2 BrendanCUDA::Fields::DField2<_T>::Dimensions() const {
    return uint32_2(lengthX, lengthY);
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::DField2<_T>::SizeOnGPU() const {
    return ((((size_t)lengthX) * ((size_t)lengthY)) * sizeof(_T)) << 1;
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::DField2<_T>::CoordinatesToIndex(uint32_2 Coordinates) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, 2, true>(Dimensions(), Coordinates);
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::DField2<_T>::CoordinatesToIndex(uint32_t X, uint32_t Y) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, 2, true>(Dimensions(), uint32_2(X, Y));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_2 BrendanCUDA::Fields::DField2<_T>::IndexToCoordinates(uint64_t Index) const {
    return BrendanCUDA::IndexToCoordinates<uint64_t, uint32_t, 3, true>(Dimensions(), Index);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::FillWith(_T Value) {
    FBack().FillWith(Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyAllIn(_T* All) {
    FBack().CopyAllIn(All);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyAllOut(_T* All) const {
    FFront().CopyAllOut(All);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint64_t Index, _T* Value) {
    FBack().CopyValueIn(Index, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint32_2 Coordinates, _T* Value) {
    FBack().CopyValueIn(Coordinates, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint32_t X, uint32_t Y, _T* Value) {
    FBack().CopyValueIn(X, Y, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
    FFront().CopyValueOut(Index, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint32_2 Coordinates, _T* Value) const {
    FFront().CopyValueOut(Coordinates, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const {
    FFront().CopyValueOut(X, Y, Value);
}
template <typename _T>
__host__ __forceinline _T* BrendanCUDA::Fields::DField2<_T>::GetAll(bool CopyToHost) const {
    return FFront().GetAll(CopyToHost);
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::Fields::DField2<_T>::GetAll() const {
    return FFront().GetAll();
}
#endif
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetAll(_T* All) {
    FBack().SetAll(All);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField2<_T>::GetValueAt(uint64_t Index) const {
    return FFront().GetValueAt(Index);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField2<_T>::GetValueAt(uint32_2 Coordinates) const {
    return FFront().GetValueAt(Coordinates);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField2<_T>::GetValueAt(uint32_t X, uint32_t Y) const {
    return FFront().GetValueAt(X, Y);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetValueAt(uint64_t Index, _T Value) {
    FBack().SetValueAt(Index, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetValueAt(uint32_2 Coordinates, _T Value) {
    FBack().SetValueAt(Coordinates, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetValueAt(uint32_t X, uint32_t Y, _T Value) {
    FBack().SetValueAt(X, Y, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyBlockIn(_T* Input, uint32_2 InputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates) {
    FBack().CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyBlockOut(_T* Output, uint32_2 OutputDimensions, uint32_2 RangeDimensions, uint32_2 RangeInInputsCoordinates, uint32_2 RangeInOutputsCoordinates) {
    FFront().CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}