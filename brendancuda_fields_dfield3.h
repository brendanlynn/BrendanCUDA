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

#ifdef __CUDACC__
            __device__ __forceinline DField3(uint32_3 Dimensions, _T* All);
            __device__ __forceinline DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All);
#endif

            __host__ __forceinline DField3(uint32_3 Dimensions, _T* All, bool CopyFromHost);
            __host__ __forceinline DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All, bool CopyFromHost);

            __host__ __device__ __forceinline uint32_t LengthX() const;
            __host__ __device__ __forceinline uint32_t LengthY() const;
            __host__ __device__ __forceinline uint32_t LengthZ() const;

            __host__ __device__ __forceinline uint32_3 Dimensions() const;

            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline Field3<_T> FFront() const;
            __host__ __device__ __forceinline Field3<_T> FBack() const;
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
            __host__ __forceinline void CopyValueIn(uint32_3 Coordinates, _T* Value, bool CopyFromHost);
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueIn(uint32_3 Coordinates, _T* Value);
#endif
            __host__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyFromHost);
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value);
#endif
            __host__ __forceinline void CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
#endif
            __host__ __forceinline void CopyValueOut(uint32_3 Coordinates, _T* Value, bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueOut(uint32_3 Coordinates, _T* Value) const;
#endif
            __host__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const;
#endif

            __host__ __forceinline _T* GetAll(bool CopyToHost) const;
#ifdef __CUDACC__
            __device__ __forceinline _T* GetAll() const;
#endif
            __host__ __device__ __forceinline void SetAll(_T* All, bool CopyFromHost);

            __host__ __device__ __forceinline _T GetValueAt(uint64_t Index) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_3 Coordinates) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const;

            __host__ __device__ __forceinline void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_3 Coordinates, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value);

            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_3 Coordinates) const;
            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ __forceinline uint32_3 IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ __forceinline _T* IndexToPointer(uint64_t Index) const;
            __host__ __device__ __forceinline uint64_t PointerToIndex(_T* Pointer) const;

            __host__ __device__ __forceinline _T* CoordinatesToPointer(uint32_3 Coordinates) const;
            __host__ __device__ __forceinline _T* CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ __forceinline uint32_3 PointerToCoordinates(_T* Pointer) const;

            __host__ __device__ __forceinline void FillWith(_T Value);

            __host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> Data() const;
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
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_3 Dimensions, _T* All)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All) { }
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All)
    : DField3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All);
}
#endif
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_3 Dimensions, _T* All, bool CopyFromHost)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::DField3<_T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All, bool CopyFromHost)
    : DField3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All, CopyFromHost);
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
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::DField3<_T>::IndexToPointer(uint64_t Index) const {
    return &cudaArrayF[Index];
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::DField3<_T>::PointerToIndex(_T* Pointer) const {
    return Pointer - cudaArrayF;
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::DField3<_T>::CoordinatesToPointer(uint32_3 Coordinates) const {
    return IndexToPointer(CoordinatesToIndex(Coordinates));
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::DField3<_T>::CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const {
    return IndexToPointer(CoordinatesToIndex(X, Y, Z));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<_T>::PointerToCoordinates(_T* Pointer) const {
    return IndexToCoordinates(PointerToIndex(Pointer));
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::FillWith(_T Value) {
    FFront().FillWith(Value);
}
template <typename _T>
__host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> BrendanCUDA::Fields::DField3<_T>::Data() const {
    return { cudaArrayF, lengthX * lengthY * lengthZ };
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyAllIn(_T* All, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(cudaArrayF, All, SizeOnGPU(), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyAllIn(_T* All) {
    deviceMemcpy(cudaArrayF, All, SizeOnGPU());
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyAllOut(_T* All, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(All, cudaArrayF, SizeOnGPU(), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyAllOut(_T* All) const {
    deviceMemcpy(All, cudaArrayF, SizeOnGPU());
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(IndexToPointer(Index), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint64_t Index, _T* Value) {
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_3 Coordinates, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_3 Coordinates, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, IndexToPointer(Index), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_3 Coordinates, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_3 Coordinates, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(_T));
}
#endif
template <typename _T>
__host__ __forceinline _T* BrendanCUDA::Fields::DField3<_T>::GetAll(bool CopyToHost) const {
    _T* a;
    if (CopyToHost) {
        a = new _T[lengthX * lengthY * lengthZ];
    }
    else {
        ThrowIfBad(cudaMalloc(&a, SizeOnGPU()));
    }
    CopyAllOut(a, CopyToHost);
    return a;
}
#ifdef __CUDACC__
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::Fields::DField3<_T>::GetAll() const {
    _T* a = new _T[lengthX * lengthY * lengthZ];
    CopyAllOut(a, false);
    return a;
}
#endif
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetAll(_T* All, bool CopyFromHost) {
    CopyAllIn(All, CopyFromHost);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint64_t Index) const {
    _T v;
#ifdef __CUDA_ARCH__
    CopyValueOut(Index, &v);
#else
    CopyValueOut(Index, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint32_3 Coordinates) const {
    return GetValueAt(Coordinates.x, Coordinates.y, Coordinates.z);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const {
    _T v;
#ifdef __CUDA_ARCH__
    CopyValueOut(X, Y, Z, &v);
#else
    CopyValueOut(X, Y, Z, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint64_t Index, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(Index, &Value);
#else
    CopyValueIn(Index, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint32_3 Coordinates, _T Value) {
    SetValueAt(Coordinates.x, Coordinates.y, Coordinates.z, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value) {
#ifdef __CUDA_ARCH__
    CopyValueIn(X, Y, Z, &Value);
#else
    CopyValueIn(X, Y, Z, &Value, true);
#endif
}