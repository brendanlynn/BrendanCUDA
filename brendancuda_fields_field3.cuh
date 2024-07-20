#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/device_ptr.h>

#include "brendancuda_dcopy.cuh"
#include "brendancuda_points.h"
#include "brendancuda_errorhelp.h"
#include "brendancuda_cudaconstexpr.h"

namespace BrendanCUDA {
#ifndef DEFINED_BrendanCUDA__details__fillWithKernel
#define DEFINED_BrendanCUDA__details__fillWithKernel
    namespace details {
        template <typename _T>
        __global__ void fillWithKernel(_T* arr, _T Value) {
            arr[blockIdx.x] = Value;
        }
    }
#endif
    namespace Fields {
        template <typename _T>
        class Field3 final {
        public:
            __host__ __device__ __forceinline Field3(uint32_3 Dimensions);
            __host__ __device__ __forceinline Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ);

            __device__ __forceinline Field3(uint32_3 Dimensions, _T* All);
            __device__ __forceinline Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All);

            __host__ __forceinline Field3(uint32_3 Dimensions, _T* All, bool CopyFromHost);
            __host__ __forceinline Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All, bool CopyFromHost);

            __host__ __device__ __forceinline uint32_t LengthX() const;
            __host__ __device__ __forceinline uint32_t LengthY() const;
            __host__ __device__ __forceinline uint32_t LengthZ() const;

            __host__ __device__ __forceinline uint32_3 Dimensions() const;

            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline std::conditional_t<isCuda, _T&, thrust::device_reference<_T>> operator()(uint32_t X, uint32_t Y, uint32_t Z);

            __host__ __forceinline void CopyAllIn(_T* All, bool CopyFromHost);
            __device__ __forceinline void CopyAllIn(_T* All);
            __host__ __forceinline void CopyAllOut(_T* All, bool CopyToHost) const;
            __device__ __forceinline void CopyAllOut(_T* All) const;
            __host__ __forceinline void CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost);
            __device__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
            __host__ __forceinline void CopyValueIn(uint32_3 Coordinates, _T* Value, bool CopyFromHost);
            __device__ __forceinline void CopyValueIn(uint32_3 Coordinates, _T* Value);
            __host__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyFromHost);
            __device__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value);
            __host__ __forceinline void CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const;
            __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ __forceinline void CopyValueOut(uint32_3 Coordinates, _T* Value, bool CopyToHost) const;
            __device__ __forceinline void CopyValueOut(uint32_3 Coordinates, _T* Value) const;
            __host__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyToHost) const;
            __device__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const;

            __host__ __forceinline _T* GetAll(bool CopyToHost) const;
            __device__ __forceinline _T* GetAll() const;
            __host__ __device__ __forceinline void SetAll(_T* All, bool CopyFromHost);

            __host__ __device__ __forceinline _T GetValueAt(uint64_t Index) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_3 Coordinates) const;
            __host__ __device__ __forceinline _T GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const;

            __host__ __device__ __forceinline void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_3 Coordinates, _T Value);
            __host__ __device__ __forceinline void SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value);

            __host__ __device__ __forceinline void Dispose();

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

            _T* cudaArray;
        };
    }
}

template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::FillWith(_T Value) {
    details::fillWithKernel<_T><<<lengthX * lengthY * lengthZ, 1>>>(cudaArray, Value);
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field3<_T>::Field3(uint32_3 Dimensions)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z) { }
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field3<_T>::Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ) {
    if (LengthX == 0 || LengthY == 0 || LengthZ == 0) {
        lengthX = 0;
        lengthY = 0;
        lengthZ = 0;
        cudaArray = 0;
    }
    else {
        lengthX = LengthX;
        lengthY = LengthY;
        lengthZ = LengthZ;
#if __CUDA_ARCH__
        cudaArray = new _T[LengthX * LengthY * LengthZ];
#else
        ThrowIfBad(cudaMalloc(&cudaArray, SizeOnGPU()));
#endif
    }
}
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::Field3<_T>::Field3(uint32_3 Dimensions, _T* All)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::Field3<_T>::Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All)
    : Field3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All);
}
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::Field3<_T>::Field3(uint32_3 Dimensions, _T* All, bool CopyFromHost)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::Field3<_T>::Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All, bool CopyFromHost)
    : Field3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All, CopyFromHost);
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field3<_T>::LengthX() const {
    return lengthX;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field3<_T>::LengthY() const {
    return lengthY;
}
template <typename _T>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::Field3<_T>::LengthZ() const {
    return lengthZ;
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_3 BrendanCUDA::Fields::Field3<_T>::Dimensions() const {
    return uint32_3(lengthX, lengthY, lengthZ);
}
template <typename _T>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::Field3<_T>::SizeOnGPU() const {
    return (((size_t)lengthX) * ((size_t)lengthY) * ((size_t)lengthZ)) * sizeof(_T);
}
template <typename _T>
__host__ __device__ __forceinline std::conditional_t<BrendanCUDA::isCuda, _T&, thrust::device_reference<_T>> BrendanCUDA::Fields::Field3<_T>::operator()(uint32_t X, uint32_t Y, uint32_t Z) {
    uint64_t idx = Coordinates32_3ToIndex64_CM(Dimensions(), uint32_3(X, Y, Z));
#ifdef __CUDA_ARCH__
    return cudaArray[idx];
#else
    return *thrust::device_ptr<_T>(cudaArray + idx);
#endif
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyAllIn(_T* All, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(cudaArray, All, SizeOnGPU(), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyAllIn(_T* All) {
    deviceMemcpy(cudaArray, All, SizeOnGPU());
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyAllOut(_T* All, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(All, cudaArray, SizeOnGPU(), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyAllOut(_T* All) const {
    deviceMemcpy(All, cudaArray, SizeOnGPU());
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(IndexToPointer(Index), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueIn(uint64_t Index, _T* Value) {
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueIn(uint32_3 Coordinates, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueIn(uint32_3 Coordinates, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, IndexToPointer(Index), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueOut(uint32_3 Coordinates, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueOut(uint32_3 Coordinates, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(_T));
}
template <typename _T>
__host__ __forceinline _T* BrendanCUDA::Fields::Field3<_T>::GetAll(bool CopyToHost) const {
    _T* a;
    if (CopyToHost) {
        a = new _T[lengthX * lengthY * lengthZ * lengthD];
    }
    else {
        ThrowIfBad(cudaMalloc(&a, SizeOnGPU()));
    }
    CopyAllOut(a, CopyToHost);
    return a;
}
template <typename _T>
__device__ __forceinline _T* BrendanCUDA::Fields::Field3<_T>::GetAll() const {
    _T* a = new _T[lengthX * lengthY * lengthZ * lengthD];
    CopyAllOut(a, false);
    return a;
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field3<_T>::SetAll(_T* All, bool CopyFromHost) {
    CopyAllIn(All, CopyFromHost);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field3<_T>::GetValueAt(uint64_t Index) const {
    _T v;
#if __CUDA_ARCH__
    CopyValueOut(Index, &v);
#else
    CopyValueOut(Index, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field3<_T>::GetValueAt(uint32_3 Coordinates) const {
    return GetValueAt(Coordinates.x, Coordinates.y, Coordinates.z);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field3<_T>::GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const {
    _T v;
#if __CUDA_ARCH__
    CopyValueOut(X, Y, Z, &v);
#else
    CopyValueOut(X, Y, Z, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::SetValueAt(uint64_t Index, _T Value) {
#if __CUDA_ARCH__
    CopyValueIn(Index, &Value);
#else
    CopyValueIn(Index, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::SetValueAt(uint32_3 Coordinates, _T Value) {
    SetValueAt(Coordinates.x, Coordinates.y, Coordinates.z, Value);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value) {
#if __CUDA_ARCH__
    CopyValueIn(X, Y, Z, &Value);
#else
    CopyValueIn(X, Y, Z, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::Dispose() {
#if __CUDA_ARCH__
    delete[] cudaArray;
#else
    ThrowIfBad(cudaFree(cudaArray));
#endif
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field3<_T>::CoordinatesToIndex(uint32_3 Coordinates) const {
    return Coordinates32_3ToIndex64_RM(Dimensions(), Coordinates);
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field3<_T>::CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const {
    return Coordinates32_3ToIndex64_RM(Dimensions(), uint32_3(X, Y, Z));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_3 BrendanCUDA::Fields::Field3<_T>::IndexToCoordinates(uint64_t Index) const {
    return Index64ToCoordinates32_3_RM(Dimensions(), Index);
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field3<_T>::IndexToPointer(uint64_t Index) const {
    return &cudaArray[Index];
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field3<_T>::PointerToIndex(_T* Pointer) const {
    return Pointer - cudaArray;
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field3<_T>::CoordinatesToPointer(uint32_3 Coordinates) const {
    return IndexToPointer(CoordinatesToIndex(Coordinates));
}
template <typename _T>
__host__ __device__ __forceinline _T* BrendanCUDA::Fields::Field3<_T>::CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const {
    return IndexToPointer(CoordinatesToIndex(X, Y, Z));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_3 BrendanCUDA::Fields::Field3<_T>::PointerToCoordinates(_T* Pointer) const {
    return IndexToCoordinates(PointerToIndex(Pointer));
}
template <typename _T>
__host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> BrendanCUDA::Fields::Field3<_T>::Data() const {
    return { thrust::device_ptr<_T>(cudaArray), (size_t)lengthX * (size_t)lengthY * (size_t)lengthZ };
}