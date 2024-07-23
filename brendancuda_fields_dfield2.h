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

#ifdef __CUDACC__
            __device__ __forceinline DField2(uint32_2 Dimensions, _T* All);
            __device__ __forceinline DField2(uint32_t LengthX, uint32_t LengthY, _T* All);
#endif

            __host__ __forceinline DField2(uint32_2 Dimensions, _T* All, bool CopyFromHost);
            __host__ __forceinline DField2(uint32_t LengthX, uint32_t LengthY, _T* All, bool CopyFromHost);

            __host__ __device__ __forceinline uint32_t LengthX() const;
            __host__ __device__ __forceinline uint32_t LengthY() const;

            __host__ __device__ __forceinline uint32_2 Dimensions() const;

            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline Field2<_T> FFront() const;
            __host__ __device__ __forceinline Field2<_T> FBack() const;
            __host__ __device__ __forceinline void Reverse();

            __host__ __forceinline void CopyAllIn(_T* All, bool CopyFromHost);
#ifdef __CUDACC__
            __device__ __forceinline void CopyAllIn(_T* All);
#endif
            __host__ __forceinline void CopyAllOut(_T* All, bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyAllOut(_T* All) const;
#endif
            __host__ __forceinline void CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost);
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
#endif
            __host__ __forceinline void CopyValueIn(uint32_2 Coordinates, _T* Value, bool CopyFromHost);
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueIn(uint32_2 Coordinates, _T* Value);
#endif
            __host__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, _T* Value, bool CopyFromHost);
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, _T* Value);
#endif
            __host__ __forceinline void CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
#endif
            __host__ __forceinline void CopyValueOut(uint32_2 Coordinates, _T* Value, bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueOut(uint32_2 Coordinates, _T* Value) const;
#endif
            __host__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, _T* Value, bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const;
#endif

            __host__ __forceinline _T* GetAll(bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline _T* GetAll() const;
#endif
            __host__ __device__ __forceinline void SetAll(_T* All, bool CopyFromHost);

            __host__ __device__ __forceinline _T GetValueAt(uint64_t Index) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_2 Coordinates) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_t X, uint32_t Y) const;

            __host__ __device__ __forceinline void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_2 Coordinates, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_t X, uint32_t Y, _T Value);

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
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_2 Dimensions, _T* All)
    : DField2(Dimensions.x, Dimensions.y, All) { }
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_t LengthX, uint32_t LengthY, _T* All)
    : DField2(LengthX, LengthY) {
    CopyAllIn(All);
}
#endif
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_2 Dimensions, _T* All, bool CopyFromHost)
    : DField2(Dimensions.x, Dimensions.y, All, CopyFromHost) { }
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::DField2<_T>::DField2(uint32_t LengthX, uint32_t LengthY, _T* All, bool CopyFromHost)
    : DField2(LengthX, LengthY) {
    CopyAllIn(All, CopyFromHost);
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
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::DField2<_T>::IndexToPointer(uint64_t Index) const {
    return &cudaArrayF[Index];
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::DField2<_T>::PointerToIndex(_T* Pointer) const {
    return Pointer - cudaArrayF;
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::DField2<_T>::CoordinatesToPointer(uint32_2 Coordinates) const {
    return IndexToPointer(CoordinatesToIndex(Coordinates));
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::DField2<_T>::CoordinatesToPointer(uint32_t X, uint32_t Y) const {
    return IndexToPointer(CoordinatesToIndex(X, Y));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_2 BrendanCUDA::Fields::DField2<_T>::PointerToCoordinates(_T* Pointer) const {
    return IndexToCoordinates(PointerToIndex(Pointer));
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::FillWith(_T Value) {
    details::fillWithKernel<_T><<<lengthX * lengthY, 1>>>(cudaArrayF, Value);
}
template <typename _T>
__host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> BrendanCUDA::Fields::DField2<_T>::Data() const {
    return { cudaArrayF, lengthX * lengthY };
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyAllIn(_T* All, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(cudaArrayF, All, SizeOnGPU(), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyAllIn(_T* All) {
    deviceMemcpy(cudaArrayF, All, SizeOnGPU());
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyAllOut(_T* All, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(All, cudaArrayF, SizeOnGPU(), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyAllOut(_T* All) const {
    deviceMemcpy(All, cudaArrayF, SizeOnGPU());
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(IndexToPointer(Index), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint64_t Index, _T* Value) {
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint32_2 Coordinates, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint32_2 Coordinates, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint32_t X, uint32_t Y, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(X, Y), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueIn(uint32_t X, uint32_t Y, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(X, Y), Value, sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, IndexToPointer(Index), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint32_2 Coordinates, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint32_2 Coordinates, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint32_t X, uint32_t Y, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(X, Y), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(X, Y), sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline _T* BrendanCUDA::Fields::DField2<_T>::GetAll(bool CopyToHost) const {
    _T* a;
    if (CopyToHost) {
        a = new _T[lengthX * lengthY];
    }
    else {
        ThrowIfBad(cudaMalloc(&a, SizeOnGPU()));
    }
    CopyAllOut(a, CopyToHost);
    return a;
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::Fields::DField2<_T>::GetAll() const {
    _T* a = new _T[lengthX * lengthY];
    CopyAllOut(a, false);
    return a;
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetAll(_T* All, bool CopyFromHost) {
    CopyAllIn(All, CopyFromHost);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField2<_T>::GetValueAt(uint64_t Index) const {
    _T v;
#ifdef __CUDA_ARCH__
    CopyValueOut(Index, &v);
#else
    CopyValueOut(Index, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField2<_T>::GetValueAt(uint32_2 Coordinates) const {
    return GetValueAt(Coordinates.x, Coordinates.y);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField2<_T>::GetValueAt(uint32_t X, uint32_t Y) const {
    _T v;
#ifdef __CUDA_ARCH__
    CopyValueOut(X, Y, &v);
#else
    CopyValueOut(X, Y, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetValueAt(uint64_t Index, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(Index, &Value);
#else
    CopyValueIn(Index, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetValueAt(uint32_2 Coordinates, _T Value) {
    SetValueAt(Coordinates.x, Coordinates.y, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField2<_T>::SetValueAt(uint32_t X, uint32_t Y, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(X, Y, &Value);
#else
    CopyValueIn(X, Y, &Value, true);
#endif
}