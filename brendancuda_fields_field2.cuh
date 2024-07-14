#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/device_ptr.h>

#include "brendancuda_dcopy.cuh"
#include "brendancuda_points.h"
#include "brendancuda_errorhelp.h"

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
        class Field2 final {
        public:
            __host__ __device__ __forceinline Field2(uint32_2 Dimensions);
            __host__ __device__ __forceinline Field2(uint32_t LengthX, uint32_t LengthY);

            __device__ __forceinline Field2(uint32_2 Dimensions, _T* All);
            __device__ __forceinline Field2(uint32_t LengthX, uint32_t LengthY, _T* All);

            __host__ __forceinline Field2(uint32_2 Dimensions, _T* All, bool CopyFromHost);
            __host__ __forceinline Field2(uint32_t LengthX, uint32_t LengthY, _T* All, bool CopyFromHost);

            __host__ __device__ __forceinline uint32_t LengthX() const;
            __host__ __device__ __forceinline uint32_t LengthY() const;

            __host__ __device__ __forceinline uint32_2 Dimensions() const;

            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __forceinline void CopyAllIn(_T* All, bool CopyFromHost);
            __device__ __forceinline void CopyAllIn(_T* All);
            __host__ __forceinline void CopyAllOut(_T* All, bool CopyToHost) const;
            __device__ __forceinline void CopyAllOut(_T* All) const;
            __host__ __forceinline void CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost);
            __device__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
            __host__ __forceinline void CopyValueIn(uint32_2 Coordinates, _T* Value, bool CopyFromHost);
            __device__ __forceinline void CopyValueIn(uint32_2 Coordinates, _T* Value);
            __host__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, _T* Value, bool CopyFromHost);
            __device__ __forceinline void CopyValueIn(uint32_t X, uint32_t Y, _T* Value);
            __host__ __forceinline void CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const;
            __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ __forceinline void CopyValueOut(uint32_2 Coordinates, _T* Value, bool CopyToHost) const;
            __device__ __forceinline void CopyValueOut(uint32_2 Coordinates, _T* Value) const;
            __host__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, _T* Value, bool CopyToHost) const;
            __device__ __forceinline void CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const;

            __host__ __forceinline _T* GetAll(bool CopyToHost) const;
            __device__ __forceinline _T* GetAll() const;
            __host__ __device__ __forceinline void SetAll(_T* All, bool CopyFromHost);

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

            __host__ __device__ __forceinline void GetConsecutives(uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(uint64_t Index, _T*& PO, _T*& NO, _T*& OP, _T*& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(uint64_t Index, _T*& PP, _T*& OP, _T*& NP, _T*& PO, _T*& NO, _T*& PN, _T*& ON, _T*& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(uint64_t Index, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(uint64_t Index, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(uint32_2 Coordinates, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(uint32_2 Coordinates, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(uint32_2 Coordinates, _T*& PO, _T*& NO, _T*& OP, _T*& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(uint32_2 Coordinates, _T*& PP, _T*& OP, _T*& NP, _T*& PO, _T*& NO, _T*& PN, _T*& ON, _T*& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(uint32_2 Coordinates, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(uint32_2 Coordinates, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(_T*& Pointer, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(_T*& Pointer, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(_T*& Pointer, _T*& PO, _T*& NO, _T*& OP, _T*& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(_T*& Pointer, _T*& PP, _T*& OP, _T*& NP, _T*& PO, _T*& NO, _T*& PN, _T*& ON, _T*& NN) const;

            __host__ __device__ __forceinline void GetConsecutives(_T*& Pointer, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) const;
            __host__ __device__ __forceinline void GetConsecutives(_T*& Pointer, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) const;

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
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::FillWith(_T Value) {
    details::fillWithKernel<_T><<<lengthX * lengthY * lengthZ, 1>>>(cudaArray, Value);
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
#if __CUDA_ARCH__
        cudaArray = new _T[LengthX * LengthY];
#else
        ThrowIfBad(cudaMalloc(&cudaArray, SizeOnGPU()));
#endif
    }
}
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_2 Dimensions, _T* All)
    : Field2(Dimensions.x, Dimensions.y, All, CopyFromHost) { }
template <typename _T>
__device__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_t LengthX, uint32_t LengthY, _T* All)
    : Field2(LengthX, LengthY) {
    CopyAllIn(All);
}
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_2 Dimensions, _T* All, bool CopyFromHost)
    : Field2(Dimensions.x, Dimensions.y, All, CopyFromHost) { }
template <typename _T>
__host__ __forceinline BrendanCUDA::Fields::Field2<_T>::Field2(uint32_t LengthX, uint32_t LengthY, _T* All, bool CopyFromHost)
    : Field2(LengthX, LengthY) {
    CopyAllIn(All, CopyFromHost);
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
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyAllIn(_T* All, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(cudaArray, All, SizeOnGPU(), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyAllIn(_T* All) {
    deviceMemcpy(cudaArray, All, SizeOnGPU());
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyAllOut(_T* All, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(All, cudaArray, SizeOnGPU(), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyAllOut(_T* All) const {
    deviceMemcpy(All, cudaArray, SizeOnGPU());
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(IndexToPointer(Index), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint64_t Index, _T* Value) {
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint32_2 Coordinates, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint32_2 Coordinates, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint32_t X, uint32_t Y, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(X, Y), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueIn(uint32_t X, uint32_t Y, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(X, Y), Value, sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, IndexToPointer(Index), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint32_2 Coordinates, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint32_2 Coordinates, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T));
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint32_t X, uint32_t Y, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(X, Y), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::CopyValueOut(uint32_t X, uint32_t Y, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(X, Y), sizeof(_T));
}
template <typename _T>
__host__ __forceinline _T* BrendanCUDA::Fields::Field2<_T>::GetAll(bool CopyToHost) const {
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
__device__ __forceinline _T* BrendanCUDA::Fields::Field2<_T>::GetAll() const {
    _T* a = new _T[lengthX * lengthY * lengthZ * lengthD];
    CopyAllOut(a, false);
    return a;
}
template <typename _T>
__host__ __forceinline void BrendanCUDA::Fields::Field2<_T>::SetAll(_T* All, bool CopyFromHost) {
    CopyAllIn(All, CopyFromHost);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field2<_T>::GetValueAt(uint64_t Index) const {
    _T v;
#if __CUDA_ARCH__
    CopyValueOut(Index, &v);
#else
    CopyValueOut(Index, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field2<_T>::GetValueAt(uint32_2 Coordinates) const {
    return GetValueAt(Coordinates.x, Coordinates.y);
}
template <typename _T>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::Field2<_T>::GetValueAt(uint32_t X, uint32_t Y) const {
    _T v;
#if __CUDA_ARCH__
    CopyValueOut(X, Y, &v);
#else
    CopyValueOut(X, Y, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::SetValueAt(uint64_t Index, _T Value) {
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
    CopyValueIn(X, Y, &Value);
#else
    CopyValueIn(X, Y, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::Dispose() {
#if __CUDA_ARCH__
    delete[] cudaArray;
#else
    ThrowIfBad(cudaFree(cudaArray));
#endif
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field2<_T>::CoordinatesToIndex(uint32_2 Coordinates) const {
    return Coordinates32_2ToIndex64_RM(Dimensions(), Coordinates);
}
template <typename _T>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::Field2<_T>::CoordinatesToIndex(uint32_t X, uint32_t Y) const {
    return Coordinates32_2ToIndex64_RM(Dimensions(), uint32_2(X, Y));
}
template <typename _T>
__host__ __device__ __forceinline BrendanCUDA::uint32_2 BrendanCUDA::Fields::Field2<_T>::IndexToCoordinates(uint64_t Index) const {
    return Index64ToCoordinates32_2_RM(Dimensions(), Index);
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
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) const {
    GetConsecutives2_RM(Dimensions(), Index, PO, NO, OP, ON);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) const {
    GetConsecutives2_RM(Dimensions(), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint64_t Index, _T*& PO, _T*& NO, _T*& OP, _T*& ON) const {
    uint64_t iPO;
    uint64_t iNO;
    uint64_t iOP;
    uint64_t iON;

    GetConsecutives2_RM(Dimensions(), Index, iPO, iNO, iOP, iON);

    PO = cudaArray + iPO;
    NO = cudaArray + iNO;
    OP = cudaArray + iOP;
    ON = cudaArray + iON;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint64_t Index, _T*& PP, _T*& OP, _T*& NP, _T*& PO, _T*& NO, _T*& PN, _T*& ON, _T*& NN) const {
    uint64_t iPP;
    uint64_t iOP;
    uint64_t iNP;
    uint64_t iPO;
    uint64_t iOO;
    uint64_t iNO;
    uint64_t iPN;
    uint64_t iON;
    uint64_t iNN;

    GetConsecutives2_RM(Dimensions(), Index, iPP, iOP, iNP, iPO, iNO, iPN, iON, iNN);

    PP = cudaArray + iPP;
    OP = cudaArray + iOP;
    NP = cudaArray + iNP;
    PO = cudaArray + iPO;
    NO = cudaArray + iNO;
    PN = cudaArray + iPN;
    ON = cudaArray + iON;
    NN = cudaArray + iNN;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint64_t Index, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) const {
    GetConsecutives2_RM(Dimensions(), Index, PO, NO, OP, ON);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint64_t Index, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) const {
    GetConsecutives2_RM(Dimensions(), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint32_2 Coordinates, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) const {
    GetConsecutives2_RM(Dimensions(), Coordinates, PO, NO, OP, ON);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint32_2 Coordinates, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) const {
    GetConsecutives2_RM(Dimensions(), Coordinates, PP, OP, NP, PO, NO, PN, ON, NN);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint32_2 Coordinates, _T*& PO, _T*& NO, _T*& OP, _T*& ON) const {
    uint64_t iPO;
    uint64_t iNO;
    uint64_t iOP;
    uint64_t iON;

    GetConsecutives2_RM(Dimensions(), Coordinates, iPO, iNO, iOP, iON);

    PO = cudaArray + iPO;
    NO = cudaArray + iNO;
    OP = cudaArray + iOP;
    ON = cudaArray + iON;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint32_2 Coordinates, _T*& PP, _T*& OP, _T*& NP, _T*& PO, _T*& NO, _T*& PN, _T*& ON, _T*& NN) const {
    uint64_t iPP;
    uint64_t iOP;
    uint64_t iNP;
    uint64_t iPO;
    uint64_t iNO;
    uint64_t iPN;
    uint64_t iON;
    uint64_t iNN;

    GetConsecutives2_RM(Dimensions(), Coordinates, iPP, iOP, iNP, iPO, iNO, iPN, iON, iNN);

    PP = cudaArray + iPP;
    OP = cudaArray + iOP;
    NP = cudaArray + iNP;
    PO = cudaArray + iPO;
    NO = cudaArray + iNO;
    PN = cudaArray + iPN;
    ON = cudaArray + iON;
    NN = cudaArray + iNN;
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint32_2 Coordinates, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) const {
    GetConsecutives2(Dimensions(), Coordinates, PO, NO, OP, ON);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(uint32_2 Coordinates, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) const {
    GetConsecutives2(Dimensions(), Coordinates, PP, OP, NP, PO, NO, PN, ON, NN);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(_T*& Pointer, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) const {
    GetConsecutives2(Dimensions(), PointerToIndex(Pointer), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(_T*& Pointer, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) const {
    GetConsecutives2(Dimensions(), PointerToIndex(Pointer), PP, OP, NP, PO, NO, PN, ON, NN);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(_T*& Pointer, _T*& PO, _T*& NO, _T*& OP, _T*& ON) const {
    GetConsecutives2(Dimensions(), PointerToIndex(Pointer), PO, NO, OP, ON);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(_T*& Pointer, _T*& PP, _T*& OP, _T*& NP, _T*& PO, _T*& NO, _T*& PN, _T*& ON, _T*& NN) const {
    GetConsecutives2(Dimensions(), PointerToIndex(Pointer), PP, OP, NP, PO, NO, PN, ON, NN);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(_T*& Pointer, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) const {
    GetConsecutives2(Dimensions(), PointerToCoordinates(Pointer), PO, NO, OP, ON);
}
template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::GetConsecutives(_T*& Pointer, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) const {
    GetConsecutives2(Dimensions(), PointerToCoordinates(Pointer), PP, OP, NP, PO, NO, PN, ON, NN);
}
template <typename _T>
__host__ __device__ __forceinline std::pair<thrust::device_ptr<_T>, size_t> BrendanCUDA::Fields::Field2<_T>::Data() const {
    return { thrust::device_ptr<_T>(cudaArray), (size_t)lengthX * (size_t)lengthY };
}