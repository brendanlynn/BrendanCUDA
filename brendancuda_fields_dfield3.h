#pragma once

#include "brendancuda_fields_field3.h"
#include "brendancuda_points.h"
#include "brendancuda_errorhelp.h"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <typename _T>
        class DField3 final {
        public:
            __host__ __device__ __forceinline DField3(uint32_3 Dimensions);
            __host__ __device__ __forceinline DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ);

            __host__ __device__ __forceinline DField3(uint32_3 Dimensions, _T* All);
            __host__ __device__ __forceinline DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All);

            __host__ __device__ __forceinline uint32_t LengthX() const;
            __host__ __device__ __forceinline uint32_t LengthY() const;
            __host__ __device__ __forceinline uint32_t LengthZ() const;

            __host__ __device__ __forceinline uint32_3 Dimensions() const;

            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline Field3<_T> FFront() const;
            __host__ __device__ __forceinline Field3<_T> FBack() const;
            __host__ __device__ __forceinline void Reverse();

            __host__ __device__ __forceinline void CopyAllIn(_T* All);
            __host__ __device__ __forceinline void CopyAllOut(_T* All) const;
            __host__ __device__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
            __host__ __device__ __forceinline void CopyValueIn(uint32_3 Coordinates, _T* Value);
            __host__ __device__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value);
            __host__ __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ __device__ __forceinline void CopyValueOut(uint32_3 Coordinates, _T* Value) const;
            __host__ __device__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const;

            template <bool _CopyToHost>
            __host__ __forceinline _T* GetAll() const;
#ifdef __CUDACC__
            __device__ __forceinline _T* GetAll() const;
#endif
            __host__ __device__ __forceinline void SetAll(_T* All);

            __host__ __device__ __forceinline _T GetValueAt(uint64_t Index) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_3 Coordinates) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const;

            __host__ __device__ __forceinline void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_3 Coordinates, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value);

            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_3 Coordinates) const;
            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ __forceinline uint32_3 IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ __forceinline void FillWith(_T Value);

            __host__ __device__ __forceinline void CopyBlockIn(_T* Input, uint32_3 InputDimensions, uint32_3 RangeDimensions, uint32_3 RangeInInputsCoordinates, uint32_3 RangeInOutputsCoordinates);
            __host__ __device__ __forceinline void CopyBlockOut(_T* Output, uint32_3 OutputDimensions, uint32_3 RangeDimensions, uint32_3 RangeInInputsCoordinates, uint32_3 RangeInOutputsCoordinates);

            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T>;
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T>;
            static __forceinline DField3<_T> Deserialize(const void*& Data) requires BSerializer::Serializable<_T>;
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T>;
        private:
            uint32_t lengthX;
            uint32_t lengthY;
            uint32_t lengthZ;

            _T* cudaArrayF;
            _T* cudaArrayB;
        };
    }
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_3 Dimensions)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z) { }
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ) {
    if (LengthX == 0 || LengthY == 0 || LengthZ == 0) {
        lengthX = 0;
        lengthY = 0;
        lengthZ = 0;
        cudaArrayF = 0;
        cudaArrayB = 0;
    }
    else {
        lengthX = LengthX;
        lengthY = LengthY;
        lengthZ = LengthZ;
#ifdef __CUDA_ARCH__
        size_t l = (size_t)LengthX * (size_t)LengthY * (size_t)LengthZ;
        cudaArrayF = new _T[l];
        cudaArrayB = new _T[l];
#else
        size_t l = (size_t)LengthX * (size_t)LengthY * (size_t)LengthZ * sizeof(_T);
        ThrowIfBad(cudaMalloc(&cudaArrayF, l));
        ThrowIfBad(cudaMalloc(&cudaArrayB, l));
#endif
    }
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_3 Dimensions, _T* All)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All) { }
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All)
    : DField3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::Dispose() {
#ifdef __CUDA_ARCH__
    delete[] cudaArrayF;
    delete[] cudaArrayB;
#else
    ThrowIfBad(cudaFree(cudaArrayF));
    ThrowIfBad(cudaFree(cudaArrayB));
#endif
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field3<_T> BrendanCUDA::Fields::DField3<_T>::FFront() const{
    return *(Field3<_T>*)this;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field3<_T> BrendanCUDA::Fields::DField3<_T>::FBack() const {
    uint8_t r[sizeof(Field3<_T>)];
    *(uint64_t*)r = *(uint64_t*)this;
    ((uint32_t*)r)[2] = ((uint32_t*)this)[2];
    ((void**)r)[2] = ((void**)this)[3];
    return *(Field3<_T>*)&r;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::Reverse() {
    _T* i = cudaArrayF;
    cudaArrayF = cudaArrayB;
    cudaArrayB = i;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField3<_T>::LengthX() const {
    return lengthX;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField3<_T>::LengthY() const {
    return lengthY;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField3<_T>::LengthZ() const {
    return lengthZ;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<_T>::Dimensions() const {
    return uint32_3(lengthX, lengthY, lengthZ);
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::DField3<_T>::SizeOnGPU() const {
    return ((((size_t)lengthX) * ((size_t)lengthY) * ((size_t)lengthZ)) * sizeof(_T)) << 1;
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::DField3<_T>::CoordinatesToIndex(uint32_3 Coordinates) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, 3, true>(Dimensions(), Coordinates);
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::DField3<_T>::CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, 3, true>(Dimensions(), uint32_3(X, Y, Z));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<_T>::IndexToCoordinates(uint64_t Index) const {
    return BrendanCUDA::IndexToCoordinates<uint64_t, uint32_t, 3, true>(Dimensions(), Index);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::FillWith(_T Value) {
    FBack().FillWith(Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyAllIn(_T* All) {
    FBack().CopyAllIn(All);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyAllOut(_T* All) const {
    FFront().CopyAllOut(All);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint64_t Index, _T* Value) {
    FBack().CopyValueIn(Index, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_3 Coordinates, _T* Value) {
    FBack().CopyValueIn(Coordinates, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) {
    FBack().CopyValueIn(X, Y, Z, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
    FFront().CopyValueOut(Index, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_3 Coordinates, _T* Value) const {
    FFront().CopyValueOut(Coordinates, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const {
    FFront().CopyValueOut(X, Y, Z, Value);
}
template <typename _T>
template <bool _CopyToHost>
__host__ __forceinline _T* BrendanCUDA::Fields::DField3<_T>::GetAll() const {
    return FFront().GetAll<_CopyToHost>();
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::Fields::DField3<_T>::GetAll() const {
    return FFront().GetAll();
}
#endif
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetAll(_T* All) {
    FBack().SetAll(All);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint64_t Index) const {
    return FFront().GetValueAt(Index);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint32_3 Coordinates) const {
    return FFront().GetValueAt(Coordinates);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const {
    return FFront().GetValueAt(X, Y, Z);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint64_t Index, _T Value) {
    FBack().SetValueAt(Index, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint32_3 Coordinates, _T Value) {
    FBack().SetValueAt(Coordinates, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value) {
    FBack().SetValueAt(X, Y, Z, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyBlockIn(_T* Input, uint32_3 InputDimensions, uint32_3 RangeDimensions, uint32_3 RangeInInputsCoordinates, uint32_3 RangeInOutputsCoordinates) {
    FBack().CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyBlockOut(_T* Output, uint32_3 OutputDimensions, uint32_3 RangeDimensions, uint32_3 RangeInInputsCoordinates, uint32_3 RangeInOutputsCoordinates) {
    FFront().CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T>
__forceinline size_t BrendanCUDA::Fields::DField3<_T>::SerializedSize() const requires BSerializer::Serializable<_T> {
    return FFront().SerializedSize();
}
template <typename _T>
__forceinline void BrendanCUDA::Fields::DField3<_T>::Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
    FFront().Serialize(Data);
}
template <typename _T>
__forceinline auto BrendanCUDA::Fields::DField3<_T>::Deserialize(const void*& Data) -> DField3<_T> requires BSerializer::Serializable<_T> {
    uint32_t lengthX = BSerializer::Deserialize<uint32_t>(Data);
    uint32_t lengthY = BSerializer::Deserialize<uint32_t>(Data);
    uint32_t lengthZ = BSerializer::Deserialize<uint32_t>(Data);
    DField3<_T> dfield(lengthX, lengthY, lengthZ);
    Field3<_T> field = dfield.FFront();
    size_t l = lengthX * lengthY * lengthZ;
    for (size_t i = 0; i < l; ++i)
        field.SetValueAt(i, BSerializer::Deserialize<_T>(Data));
    return field;
}
template <typename _T>
__forceinline void BrendanCUDA::Fields::DField3<_T>::Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
    new (Value) DField3<_T>(Deserialize(Data));
}