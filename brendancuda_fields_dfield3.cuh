#pragma once

#include "brendancuda_fields_field3.cuh"
#include "brendancuda_fields_points.cuh"
#include "brendancuda_cudaerrorhelpers.h"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <typename _T>
        class DField3 final {
        public:
            __host__ __device__ DField3(uint32_3 Dimensions);
            __host__ __device__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ);

            __device__ DField3(uint32_3 Dimensions, _T* All);
            __device__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All);

            __host__ DField3(uint32_3 Dimensions, _T* All, bool CopyFromHost);
            __host__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All, bool CopyFromHost);

            __host__ __device__ uint32_t LengthX() const;
            __host__ __device__ uint32_t LengthY() const;
            __host__ __device__ uint32_t LengthZ() const;

            __host__ __device__ uint32_3 Dimensions() const;

            __host__ __device__ size_t SizeOnGPU() const;

            __host__ __device__ void Dispose();

            __host__ __device__ Field3<_T> FFront() const;
            __host__ __device__ Field3<_T> FBack() const;
            __host__ __device__ void Reverse();

            __host__ void CopyAllIn(_T* All, bool CopyFromHost);
            __device__ void CopyAllIn(_T* All);
            __host__ void CopyAllOut(_T* All, bool CopyToHost) const;
            __device__ void CopyAllOut(_T* All) const;
            __host__ void CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint64_t Index, _T* Value);
            __host__ void CopyValueIn(uint32_3 Coordinates, _T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint32_3 Coordinates, _T* Value);
            __host__ void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value);
            __host__ void CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ void CopyValueOut(uint32_3 Coordinates, _T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint32_3 Coordinates, _T* Value) const;
            __host__ void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const;

            __host__ _T* GetAll(bool CopyToHost) const;
            __device__ _T* GetAll() const;
            __host__ __device__ void SetAll(_T* All, bool CopyFromHost);

            __host__ __device__ _T GetValueAt(uint64_t Index) const;
            __host__ __device__ _T GetValueAt(uint32_3 Coordinates) const;
            __host__ __device__ _T GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const;

            __host__ __device__ void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ void SetValueAt(uint32_3 Coordinates, _T Value);
            __host__ __device__ void SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value);

            __host__ __device__ uint64_t CoordinatesToIndex(uint32_3 Coordinates) const;
            __host__ __device__ uint64_t CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ uint32_3 IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ _T* IndexToPointer(uint64_t Index) const;
            __host__ __device__ uint64_t PointerToIndex(_T* Pointer) const;

            __host__ __device__ _T* CoordinatesToPointer(uint32_3 Coordinates) const;
            __host__ __device__ _T* CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ uint32_3 PointerToCoordinates(_T* Pointer) const;

            __host__ __device__ void GetConsecutives(uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const;
            __host__ __device__ void GetConsecutives(uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const;

            __host__ __device__ void GetConsecutives(uint64_t Index, _T*& POO, _T*& NOO, _T*& OPO, _T*& ONO, _T*& OOP, _T*& OON) const;
            __host__ __device__ void GetConsecutives(uint64_t Index, _T*& PPP, _T*& OPP, _T*& NPP, _T*& POP, _T*& OOP, _T*& NOP, _T*& PNP, _T*& ONP, _T*& NNP, _T*& PPO, _T*& OPO, _T*& NPO, _T*& POO, _T*& NOO, _T*& PNO, _T*& ONO, _T*& NNO, _T*& PPN, _T*& OPN, _T*& NPN, _T*& PON, _T*& OON, _T*& NON, _T*& PNN, _T*& ONN, _T*& NNN) const;

            __host__ __device__ void GetConsecutives(uint64_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const;
            __host__ __device__ void GetConsecutives(uint64_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const;

            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const;
            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const;

            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, _T*& POO, _T*& NOO, _T*& OPO, _T*& ONO, _T*& OOP, _T*& OON) const;
            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, _T*& PPP, _T*& OPP, _T*& NPP, _T*& POP, _T*& OOP, _T*& NOP, _T*& PNP, _T*& ONP, _T*& NNP, _T*& PPO, _T*& OPO, _T*& NPO, _T*& POO, _T*& NOO, _T*& PNO, _T*& ONO, _T*& NNO, _T*& PPN, _T*& OPN, _T*& NPN, _T*& PON, _T*& OON, _T*& NON, _T*& PNN, _T*& ONN, _T*& NNN) const;

            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const;
            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const;

            __host__ __device__ void GetConsecutives(_T*& Pointer, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const;
            __host__ __device__ void GetConsecutives(_T*& Pointer, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const;

            __host__ __device__ void GetConsecutives(_T*& Pointer, _T*& POO, _T*& NOO, _T*& OPO, _T*& ONO, _T*& OOP, _T*& OON) const;
            __host__ __device__ void GetConsecutives(_T*& Pointer, _T*& PPP, _T*& OPP, _T*& NPP, _T*& POP, _T*& OOP, _T*& NOP, _T*& PNP, _T*& ONP, _T*& NNP, _T*& PPO, _T*& OPO, _T*& NPO, _T*& POO, _T*& NOO, _T*& PNO, _T*& ONO, _T*& NNO, _T*& PPN, _T*& OPN, _T*& NPN, _T*& PON, _T*& OON, _T*& NON, _T*& PNN, _T*& ONN, _T*& NNN) const;

            __host__ __device__ void GetConsecutives(_T*& Pointer, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const;
            __host__ __device__ void GetConsecutives(_T*& Pointer, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const;

            __host__ __device__ void FillWith(_T Value);

            __host__ __device__ std::pair<thrust::device_ptr<_T>, size_t> Data() const;
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
__host__ __device__ BrendanCUDA::Fields::DField3<_T>::DField3(uint32_3 Dimensions)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z) { }
template <typename _T>
__host__ __device__ BrendanCUDA::Fields::DField3<_T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ) {
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
#if __CUDA_ARCH__
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
__device__ BrendanCUDA::Fields::DField3<_T>::DField3(uint32_3 Dimensions, _T* All)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All) { }
template <typename _T>
__device__ BrendanCUDA::Fields::DField3<_T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All)
    : DField3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All);
}
template <typename _T>
__host__ BrendanCUDA::Fields::DField3<_T>::DField3(uint32_3 Dimensions, _T* All, bool CopyFromHost)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename _T>
__host__ BrendanCUDA::Fields::DField3<_T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, _T* All, bool CopyFromHost)
    : DField3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All, CopyFromHost);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::Dispose() {
#if __CUDA_ARCH__
    delete[] cudaArrayF;
    delete[] cudaArrayB;
#else
    ThrowIfBad(cudaFree(cudaArrayF));
    ThrowIfBad(cudaFree(cudaArrayB));
#endif
}
template <typename _T>
__host__ __device__ BrendanCUDA::Fields::Field3<_T> BrendanCUDA::Fields::DField3<_T>::FFront() const{
    return *(Field3<_T>*)this;
}
template <typename _T>
__host__ __device__ BrendanCUDA::Fields::Field3<_T> BrendanCUDA::Fields::DField3<_T>::FBack() const {
    uint8_t r[sizeof(Field3<_T>)];
    *(uint64_t*)r = *(uint64_t*)this;
    ((uint32_t*)r)[2] = ((uint32_t*)this)[2];
    ((void**)r)[2] = ((void**)this)[3];
    return *(Field3<_T>*)&r;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::Reverse() {
    _T* i = cudaArrayF;
    cudaArrayF = cudaArrayB;
    cudaArrayB = i;
}
template <typename _T>
__host__ __device__ uint32_t BrendanCUDA::Fields::DField3<_T>::LengthX() const {
    return lengthX;
}
template <typename _T>
__host__ __device__ uint32_t BrendanCUDA::Fields::DField3<_T>::LengthY() const {
    return lengthY;
}
template <typename _T>
__host__ __device__ uint32_t BrendanCUDA::Fields::DField3<_T>::LengthZ() const {
    return lengthZ;
}
template <typename _T>
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<_T>::Dimensions() const {
    return uint32_3(lengthX, lengthY, lengthZ);
}
template <typename _T>
__host__ __device__ size_t BrendanCUDA::Fields::DField3<_T>::SizeOnGPU() const {
    return ((((size_t)lengthX) * ((size_t)lengthY) * ((size_t)lengthZ)) * sizeof(_T)) << 1;
}
template <typename _T>
__host__ __device__ uint64_t BrendanCUDA::Fields::DField3<_T>::CoordinatesToIndex(uint32_3 Coordinates) const {
    return Fields::Coordinates32_3ToIndex64_RM(Dimensions(), Coordinates);
}
template <typename _T>
__host__ __device__ uint64_t BrendanCUDA::Fields::DField3<_T>::CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const {
    return Fields::Coordinates32_3ToIndex64_RM(Dimensions(), uint32_3(X, Y, Z));
}
template <typename _T>
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<_T>::IndexToCoordinates(uint64_t Index) const {
    return Fields::Index64ToCoordinates32_3_RM(Dimensions(), Index);
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::Fields::DField3<_T>::IndexToPointer(uint64_t Index) const {
    return &cudaArrayF[Index];
}
template <typename _T>
__host__ __device__ uint64_t BrendanCUDA::Fields::DField3<_T>::PointerToIndex(_T* Pointer) const {
    return Pointer - cudaArrayF;
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::Fields::DField3<_T>::CoordinatesToPointer(uint32_3 Coordinates) const {
    return IndexToPointer(CoordinatesToIndex(Coordinates));
}
template <typename _T>
__host__ __device__ _T* BrendanCUDA::Fields::DField3<_T>::CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const {
    return IndexToPointer(CoordinatesToIndex(X, Y, Z));
}
template <typename _T>
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<_T>::PointerToCoordinates(_T* Pointer) const {
    return IndexToCoordinates(PointerToIndex(Pointer));
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const {
    Fields::GetConsecutives_RM(Dimensions(), Index, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const {
    Fields::GetConsecutives_RM(Dimensions(), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint64_t Index, _T*& POO, _T*& NOO, _T*& OPO, _T*& ONO, _T*& OOP, _T*& OON) const {
    uint64_t iPOO;
    uint64_t iNOO;
    uint64_t iOPO;
    uint64_t iONO;
    uint64_t iOOP;
    uint64_t iOON;

    Fields::GetConsecutives_RM(Dimensions(), Index, iPOO, iNOO, iOPO, iONO, iOOP, iOON);

    POO = cudaArrayF + iPOO;
    NOO = cudaArrayF + iNOO;
    OPO = cudaArrayF + iOPO;
    ONO = cudaArrayF + iONO;
    OOP = cudaArrayF + iOOP;
    OON = cudaArrayF + iOON;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint64_t Index, _T*& PPP, _T*& OPP, _T*& NPP, _T*& POP, _T*& OOP, _T*& NOP, _T*& PNP, _T*& ONP, _T*& NNP, _T*& PPO, _T*& OPO, _T*& NPO, _T*& POO, _T*& NOO, _T*& PNO, _T*& ONO, _T*& NNO, _T*& PPN, _T*& OPN, _T*& NPN, _T*& PON, _T*& OON, _T*& NON, _T*& PNN, _T*& ONN, _T*& NNN) const {
    uint64_t iPPP;
    uint64_t iOPP;
    uint64_t iNPP;
    uint64_t iPOP;
    uint64_t iOOP;
    uint64_t iNOP;
    uint64_t iPNP;
    uint64_t iONP;
    uint64_t iNNP;
    uint64_t iPPO;
    uint64_t iOPO;
    uint64_t iNPO;
    uint64_t iPOO;
    uint64_t iNOO;
    uint64_t iPNO;
    uint64_t iONO;
    uint64_t iNNO;
    uint64_t iPPN;
    uint64_t iOPN;
    uint64_t iNPN;
    uint64_t iPON;
    uint64_t iOON;
    uint64_t iNON;
    uint64_t iPNN;
    uint64_t iONN;
    uint64_t iNNN;

    Fields::GetConsecutives_RM(Dimensions(), Index, iPPP, iOPP, iNPP, iPOP, iOOP, iNOP, iPNP, iONP, iNNP, iPPO, iOPO, iNPO, iPOO, iNOO, iPNO, iONO, iNNO, iPPN, iOPN, iNPN, iPON, iOON, iNON, iPNN, iONN, iNNN);

    PPP = cudaArrayF + iPPP;
    OPP = cudaArrayF + iOPP;
    NPP = cudaArrayF + iNPP;
    POP = cudaArrayF + iPOP;
    OOP = cudaArrayF + iOOP;
    NOP = cudaArrayF + iNOP;
    PNP = cudaArrayF + iPNP;
    ONP = cudaArrayF + iONP;
    NNP = cudaArrayF + iNNP;
    PPO = cudaArrayF + iPPO;
    OPO = cudaArrayF + iOPO;
    NPO = cudaArrayF + iNPO;
    POO = cudaArrayF + iPOO;
    NOO = cudaArrayF + iNOO;
    PNO = cudaArrayF + iPNO;
    ONO = cudaArrayF + iONO;
    NNO = cudaArrayF + iNNO;
    PPN = cudaArrayF + iPPN;
    OPN = cudaArrayF + iOPN;
    NPN = cudaArrayF + iNPN;
    PON = cudaArrayF + iPON;
    OON = cudaArrayF + iOON;
    NON = cudaArrayF + iNON;
    PNN = cudaArrayF + iPNN;
    ONN = cudaArrayF + iONN;
    NNN = cudaArrayF + iNNN;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint64_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const {
    Fields::GetConsecutives_RM(Dimensions(), Index, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint64_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const {
    Fields::GetConsecutives_RM(Dimensions(), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint32_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const {
    Fields::GetConsecutives_RM(Dimensions(), Coordinates, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint32_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const {
    Fields::GetConsecutives_RM(Dimensions(), Coordinates, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint32_3 Coordinates, _T*& POO, _T*& NOO, _T*& OPO, _T*& ONO, _T*& OOP, _T*& OON) const {
    uint64_t iPOO;
    uint64_t iNOO;
    uint64_t iOPO;
    uint64_t iONO;
    uint64_t iOOP;
    uint64_t iOON;

    Fields::GetConsecutives_RM(Dimensions(), Coordinates, iPOO, iNOO, iOPO, iONO, iOOP, iOON);

    POO = cudaArrayF + iPOO;
    NOO = cudaArrayF + iNOO;
    OPO = cudaArrayF + iOPO;
    ONO = cudaArrayF + iONO;
    OOP = cudaArrayF + iOOP;
    OON = cudaArrayF + iOON;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint32_3 Coordinates, _T*& PPP, _T*& OPP, _T*& NPP, _T*& POP, _T*& OOP, _T*& NOP, _T*& PNP, _T*& ONP, _T*& NNP, _T*& PPO, _T*& OPO, _T*& NPO, _T*& POO, _T*& NOO, _T*& PNO, _T*& ONO, _T*& NNO, _T*& PPN, _T*& OPN, _T*& NPN, _T*& PON, _T*& OON, _T*& NON, _T*& PNN, _T*& ONN, _T*& NNN) const {
    uint64_t iPPP;
    uint64_t iOPP;
    uint64_t iNPP;
    uint64_t iPOP;
    uint64_t iOOP;
    uint64_t iNOP;
    uint64_t iPNP;
    uint64_t iONP;
    uint64_t iNNP;
    uint64_t iPPO;
    uint64_t iOPO;
    uint64_t iNPO;
    uint64_t iPOO;
    uint64_t iNOO;
    uint64_t iPNO;
    uint64_t iONO;
    uint64_t iNNO;
    uint64_t iPPN;
    uint64_t iOPN;
    uint64_t iNPN;
    uint64_t iPON;
    uint64_t iOON;
    uint64_t iNON;
    uint64_t iPNN;
    uint64_t iONN;
    uint64_t iNNN;

    Fields::GetConsecutives_RM(Dimensions(), Coordinates, iPPP, iOPP, iNPP, iPOP, iOOP, iNOP, iPNP, iONP, iNNP, iPPO, iOPO, iNPO, iPOO, iNOO, iPNO, iONO, iNNO, iPPN, iOPN, iNPN, iPON, iOON, iNON, iPNN, iONN, iNNN);

    PPP = cudaArrayF + iPPP;
    OPP = cudaArrayF + iOPP;
    NPP = cudaArrayF + iNPP;
    POP = cudaArrayF + iPOP;
    OOP = cudaArrayF + iOOP;
    NOP = cudaArrayF + iNOP;
    PNP = cudaArrayF + iPNP;
    ONP = cudaArrayF + iONP;
    NNP = cudaArrayF + iNNP;
    PPO = cudaArrayF + iPPO;
    OPO = cudaArrayF + iOPO;
    NPO = cudaArrayF + iNPO;
    POO = cudaArrayF + iPOO;
    NOO = cudaArrayF + iNOO;
    PNO = cudaArrayF + iPNO;
    ONO = cudaArrayF + iONO;
    NNO = cudaArrayF + iNNO;
    PPN = cudaArrayF + iPPN;
    OPN = cudaArrayF + iOPN;
    NPN = cudaArrayF + iNPN;
    PON = cudaArrayF + iPON;
    OON = cudaArrayF + iOON;
    NON = cudaArrayF + iNON;
    PNN = cudaArrayF + iPNN;
    ONN = cudaArrayF + iONN;
    NNN = cudaArrayF + iNNN;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint32_3 Coordinates, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const {
    Fields::GetConsecutives(Dimensions(), Coordinates, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(uint32_3 Coordinates, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const {
    Fields::GetConsecutives(Dimensions(), Coordinates, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(_T*& Pointer, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const {
    Fields::GetConsecutives(Dimensions(), PointerToIndex(Pointer), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(_T*& Pointer, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const {
    Fields::GetConsecutives(Dimensions(), PointerToIndex(Pointer), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(_T*& Pointer, _T*& POO, _T*& NOO, _T*& OPO, _T*& ONO, _T*& OOP, _T*& OON) const {
    Fields::GetConsecutives(Dimensions(), PointerToIndex(Pointer), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(_T*& Pointer, _T*& PPP, _T*& OPP, _T*& NPP, _T*& POP, _T*& OOP, _T*& NOP, _T*& PNP, _T*& ONP, _T*& NNP, _T*& PPO, _T*& OPO, _T*& NPO, _T*& POO, _T*& NOO, _T*& PNO, _T*& ONO, _T*& NNO, _T*& PPN, _T*& OPN, _T*& NPN, _T*& PON, _T*& OON, _T*& NON, _T*& PNN, _T*& ONN, _T*& NNN) const {
    Fields::GetConsecutives(Dimensions(), PointerToIndex(Pointer), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(_T*& Pointer, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const {
    Fields::GetConsecutives(Dimensions(), PointerToCoordinates(Pointer), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::GetConsecutives(_T*& Pointer, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const {
    Fields::GetConsecutives(Dimensions(), PointerToCoordinates(Pointer), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::FillWith(_T Value) {
    fillWithKernel<_T><<<lengthX * lengthY * lengthZ, 1>>>(cudaArrayF, Value);
}
template <typename _T>
__host__ __device__ std::pair<thrust::device_ptr<_T>, size_t> BrendanCUDA::Fields::DField3<_T>::Data() const {
    return { cudaArrayF, lengthX * lengthY * lengthZ };
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyAllIn(_T* All, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(cudaArrayF, All, SizeOnGPU(), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyAllIn(_T* All) {
    deviceMemcpy(cudaArrayF, All, SizeOnGPU());
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyAllOut(_T* All, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(All, cudaArrayF, SizeOnGPU(), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyAllOut(_T* All) const {
    deviceMemcpy(All, cudaArrayF, SizeOnGPU());
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint64_t Index, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(IndexToPointer(Index), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint64_t Index, _T* Value) {
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(_T));
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_3 Coordinates, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_3 Coordinates, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(_T));
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyFromHost) {
    ThrowIfBad(cudaMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(_T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) {
    deviceMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(_T));
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint64_t Index, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, IndexToPointer(Index), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint64_t Index, _T* Value) const {
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(_T));
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_3 Coordinates, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_3 Coordinates, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(_T));
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value, bool CopyToHost) const {
    ThrowIfBad(cudaMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(_T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}
template <typename _T>
__device__ void BrendanCUDA::Fields::DField3<_T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, _T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(_T));
}
template <typename _T>
__host__ _T* BrendanCUDA::Fields::DField3<_T>::GetAll(bool CopyToHost) const {
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
__device__ _T* BrendanCUDA::Fields::DField3<_T>::GetAll() const {
    _T* a = new _T[lengthX * lengthY * lengthZ * lengthD];
    CopyAllOut(a, false);
    return a;
}
template <typename _T>
__host__ void BrendanCUDA::Fields::DField3<_T>::SetAll(_T* All, bool CopyFromHost) {
    CopyAllIn(All, CopyFromHost);
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint64_t Index) const {
    _T v;
#if __CUDA_ARCH__
    CopyValueOut(Index, &v);
#else
    CopyValueOut(Index, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint32_3 Coordinates) const {
    return GetValueAt(Coordinates.x, Coordinates.y, Coordinates.z);
}
template <typename _T>
__host__ __device__ _T BrendanCUDA::Fields::DField3<_T>::GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const {
    _T v;
#if __CUDA_ARCH__
    CopyValueOut(X, Y, Z, &v);
#else
    CopyValueOut(X, Y, Z, &v, true);
#endif
    return v;
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint64_t Index, _T Value) {
#if __CUDA_ARCH__
    CopyValueIn(Index, &Value);
#else
    CopyValueIn(Index, &Value, true);
#endif
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint32_3 Coordinates, _T Value) {
    SetValueAt(Coordinates.x, Coordinates.y, Coordinates.z, Value);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::Fields::DField3<_T>::SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, _T Value) {
#if __CUDA_ARCH__
    CopyValueIn(X, Y, Z, &Value);
#else
    CopyValueIn(X, Y, Z, &Value, true);
#endif
}