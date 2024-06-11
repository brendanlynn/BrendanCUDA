#pragma once

#include "brendancuda_fields_field3.cuh"
#include "brendancuda_fields_points.cuh"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <typename T>
        class DField3 final {
        public:
            __host__ __device__ DField3(uint32_3 Dimensions);
            __host__ __device__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ);

            __device__ DField3(uint32_3 Dimensions, T* All);
            __device__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All);

            __host__ DField3(uint32_3 Dimensions, T* All, bool CopyFromHost);
            __host__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All, bool CopyFromHost);

            __host__ __device__ uint32_t LengthX() const;
            __host__ __device__ uint32_t LengthY() const;
            __host__ __device__ uint32_t LengthZ() const;

            __host__ __device__ uint32_3 Dimensions() const;

            __host__ __device__ size_t SizeOnGPU() const;

            __host__ __device__ void Dispose();

            __host__ __device__ Field3<T> FFront() const;
            __host__ __device__ Field3<T> FBack() const;
            __host__ __device__ void Reverse();

            __host__ void CopyAllIn(T* All, bool CopyFromHost);
            __device__ void CopyAllIn(T* All);
            __host__ void CopyAllOut(T* All, bool CopyToHost) const;
            __device__ void CopyAllOut(T* All) const;
            __host__ void CopyValueIn(uint64_t Index, T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint64_t Index, T* Value);
            __host__ void CopyValueIn(uint32_3 Coordinates, T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint32_3 Coordinates, T* Value);
            __host__ void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value);
            __host__ void CopyValueOut(uint64_t Index, T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint64_t Index, T* Value) const;
            __host__ void CopyValueOut(uint32_3 Coordinates, T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint32_3 Coordinates, T* Value) const;
            __host__ void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value) const;

            __host__ T* GetAll(bool CopyToHost) const;
            __device__ T* GetAll() const;
            __host__ __device__ void SetAll(T* All, bool CopyFromHost);

            __host__ __device__ T GetValueAt(uint64_t Index) const;
            __host__ __device__ T GetValueAt(uint32_3 Coordinates) const;
            __host__ __device__ T GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const;

            __host__ __device__ void SetValueAt(uint64_t Index, T Value);
            __host__ __device__ void SetValueAt(uint32_3 Coordinates, T Value);
            __host__ __device__ void SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, T Value);

            __host__ __device__ uint64_t CoordinatesToIndex(uint32_3 Coordinates) const;
            __host__ __device__ uint64_t CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ uint32_3 IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ T* IndexToPointer(uint64_t Index) const;
            __host__ __device__ uint64_t PointerToIndex(T* Pointer) const;

            __host__ __device__ T* CoordinatesToPointer(uint32_3 Coordinates) const;
            __host__ __device__ T* CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ uint32_3 PointerToCoordinates(T* Pointer) const;

            __host__ __device__ void GetConsecutives(uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const;
            __host__ __device__ void GetConsecutives(uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const;

            __host__ __device__ void GetConsecutives(uint64_t Index, T*& POO, T*& NOO, T*& OPO, T*& ONO, T*& OOP, T*& OON) const;
            __host__ __device__ void GetConsecutives(uint64_t Index, T*& PPP, T*& OPP, T*& NPP, T*& POP, T*& OOP, T*& NOP, T*& PNP, T*& ONP, T*& NNP, T*& PPO, T*& OPO, T*& NPO, T*& POO, T*& NOO, T*& PNO, T*& ONO, T*& NNO, T*& PPN, T*& OPN, T*& NPN, T*& PON, T*& OON, T*& NON, T*& PNN, T*& ONN, T*& NNN) const;

            __host__ __device__ void GetConsecutives(uint64_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const;
            __host__ __device__ void GetConsecutives(uint64_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const;

            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const;
            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const;

            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, T*& POO, T*& NOO, T*& OPO, T*& ONO, T*& OOP, T*& OON) const;
            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, T*& PPP, T*& OPP, T*& NPP, T*& POP, T*& OOP, T*& NOP, T*& PNP, T*& ONP, T*& NNP, T*& PPO, T*& OPO, T*& NPO, T*& POO, T*& NOO, T*& PNO, T*& ONO, T*& NNO, T*& PPN, T*& OPN, T*& NPN, T*& PON, T*& OON, T*& NON, T*& PNN, T*& ONN, T*& NNN) const;

            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const;
            __host__ __device__ void GetConsecutives(uint32_3 Coordinates, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const;

            __host__ __device__ void GetConsecutives(T*& Pointer, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const;
            __host__ __device__ void GetConsecutives(T*& Pointer, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const;

            __host__ __device__ void GetConsecutives(T*& Pointer, T*& POO, T*& NOO, T*& OPO, T*& ONO, T*& OOP, T*& OON) const;
            __host__ __device__ void GetConsecutives(T*& Pointer, T*& PPP, T*& OPP, T*& NPP, T*& POP, T*& OOP, T*& NOP, T*& PNP, T*& ONP, T*& NNP, T*& PPO, T*& OPO, T*& NPO, T*& POO, T*& NOO, T*& PNO, T*& ONO, T*& NNO, T*& PPN, T*& OPN, T*& NPN, T*& PON, T*& OON, T*& NON, T*& PNN, T*& ONN, T*& NNN) const;

            __host__ __device__ void GetConsecutives(T*& Pointer, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const;
            __host__ __device__ void GetConsecutives(T*& Pointer, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const;

            __host__ __device__ void FillWith(T Value);

            __host__ __device__ std::pair<thrust::device_ptr<T>, size_t> Data() const;
        private:
            uint32_t lengthX;
            uint32_t lengthY;
            uint32_t lengthZ;

            T* cudaArrayF;
            T* cudaArrayB;
        };
    }
}
template <typename T>
__host__ __device__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_3 Dimensions)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z) { }
template <typename T>
__host__ __device__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ) {
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
        cudaArrayF = new T[l];
        cudaArrayB = new T[l];
#else
        size_t l = (size_t)LengthX * (size_t)LengthY * (size_t)LengthZ * sizeof(T);
        cudaError_t eF = cudaMalloc(&cudaArrayF, l);
        cudaError_t eB = cudaMalloc(&cudaArrayB, l);
        if (eF) {
            throw std::runtime_error("A CUDA error occured when attempting to allocate 2 lengths of " + std::to_string(l) + " bytes of VRAM memory. Error #" + std::to_string(eF) + ".");
        }
        if (eB) {
            throw std::runtime_error("A CUDA error occured when attempting to allocate 2 lengths of " + std::to_string(l) + " bytes of VRAM memory. Error #" + std::to_string(eB) + ".");
        }
#endif
    }
}
template <typename T>
__device__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_3 Dimensions, T* All)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All) { }
template <typename T>
__device__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All)
    : DField3(LengthX, LengthY, LengthZ) {
    FFront().CopyAllIn(All);
}
template <typename T>
__host__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_3 Dimensions, T* All, bool CopyFromHost)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename T>
__host__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All, bool CopyFromHost)
    : DField3(LengthX, LengthY, LengthZ) {
    FFront().CopyAllIn(All, CopyFromHost);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::Dispose() {
#if __CUDA_ARCH__
    delete[] cudaArrayF;
    delete[] cudaArrayB;
#else
    cudaFree(cudaArrayF);
    cudaFree(cudaArrayB);
#endif
}
template <typename T>
__host__ __device__ BrendanCUDA::Fields::Field3<T> BrendanCUDA::Fields::DField3<T>::FFront() const{
    return *(Field3<T>*)this;
}
template <typename T>
__host__ __device__ BrendanCUDA::Fields::Field3<T> BrendanCUDA::Fields::DField3<T>::FBack() const {
    uint8_t r[sizeof(Field3<T>)];
    *(uint64_t*)r = *(uint64_t*)this;
    ((uint32_t*)r)[2] = ((uint32_t*)this)[2];
    ((void**)r)[2] = ((void**)this)[3];
    return *(Field3<T>*)&r;
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::Reverse() {
    T* i = cudaArrayF;
    cudaArrayF = cudaArrayB;
    cudaArrayB = i;
}
template <typename T>
__host__ __device__ uint32_t BrendanCUDA::Fields::DField3<T>::LengthX() const {
    return lengthX;
}
template <typename T>
__host__ __device__ uint32_t BrendanCUDA::Fields::DField3<T>::LengthY() const {
    return lengthY;
}
template <typename T>
__host__ __device__ uint32_t BrendanCUDA::Fields::DField3<T>::LengthZ() const {
    return lengthZ;
}
template <typename T>
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<T>::Dimensions() const {
    return uint32_3(lengthX, lengthY, lengthZ);
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::Fields::DField3<T>::SizeOnGPU() const {
    return ((((size_t)lengthX) * ((size_t)lengthY) * ((size_t)lengthZ)) * sizeof(T)) << 1;
}
template <typename T>
__host__ __device__ uint64_t BrendanCUDA::Fields::DField3<T>::CoordinatesToIndex(uint32_3 Coordinates) const {
    return FFront().CoordinatesToIndex(Coordinates);
}
template <typename T>
__host__ __device__ uint64_t BrendanCUDA::Fields::DField3<T>::CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const {
    return FFront().CoordinatesToIndex(X, Y, Z);
}
template <typename T>
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<T>::IndexToCoordinates(uint64_t Index) const {
    return FFront().IndexToCoordinates(Index);
}
template <typename T>
__host__ __device__ T* BrendanCUDA::Fields::DField3<T>::IndexToPointer(uint64_t Index) const {
    return FFront().IndexToPointer(Index);
}
template <typename T>
__host__ __device__ uint64_t BrendanCUDA::Fields::DField3<T>::PointerToIndex(T* Pointer) const {
    return FFront().PointerToIndex(Pointer);
}
template <typename T>
__host__ __device__ T* BrendanCUDA::Fields::DField3<T>::CoordinatesToPointer(uint32_3 Coordinates) const {
    return FFront().CoordinatesToPointer(Coordinates);
}
template <typename T>
__host__ __device__ T* BrendanCUDA::Fields::DField3<T>::CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const {
    return FFront().CoordinatesToPointer(X, Y, Z);
}
template <typename T>
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Fields::DField3<T>::PointerToCoordinates(T* Pointer) const {
    return FFront().PointerToCoordinates(Pointer);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const {
    return FFront().GetConsecutives(Index, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const {
    return FFront().GetConsecutives(Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint64_t Index, T*& POO, T*& NOO, T*& OPO, T*& ONO, T*& OOP, T*& OON) const {
    return FFront().GetConsecutives(Index, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint64_t Index, T*& PPP, T*& OPP, T*& NPP, T*& POP, T*& OOP, T*& NOP, T*& PNP, T*& ONP, T*& NNP, T*& PPO, T*& OPO, T*& NPO, T*& POO, T*& NOO, T*& PNO, T*& ONO, T*& NNO, T*& PPN, T*& OPN, T*& NPN, T*& PON, T*& OON, T*& NON, T*& PNN, T*& ONN, T*& NNN) const {
    return FFront().GetConsecutives(Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint64_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const {
    return FFront().GetConsecutives(Index, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint64_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const {
    return FFront().GetConsecutives(Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint32_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const {
    return FFront().GetConsecutives(Coordinates, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint32_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const {
    return FFront().GetConsecutives(Coordinates, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint32_3 Coordinates, T*& POO, T*& NOO, T*& OPO, T*& ONO, T*& OOP, T*& OON) const {
    return FFront().GetConsecutives(Coordinates, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint32_3 Coordinates, T*& PPP, T*& OPP, T*& NPP, T*& POP, T*& OOP, T*& NOP, T*& PNP, T*& ONP, T*& NNP, T*& PPO, T*& OPO, T*& NPO, T*& POO, T*& NOO, T*& PNO, T*& ONO, T*& NNO, T*& PPN, T*& OPN, T*& NPN, T*& PON, T*& OON, T*& NON, T*& PNN, T*& ONN, T*& NNN) const {
    return FFront().GetConsecutives(Coordinates, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint32_3 Coordinates, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const {
    return FFront().GetConsecutives(Coordinates, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(uint32_3 Coordinates, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const {
    return FFront().GetConsecutives(Coordinates, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(T*& Pointer, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) const {
    return FFront().GetConsecutives(Pointer, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(T*& Pointer, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) const {
    return FFront().GetConsecutives(Pointer, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(T*& Pointer, T*& POO, T*& NOO, T*& OPO, T*& ONO, T*& OOP, T*& OON) const {
    return FFront().GetConsecutives(Pointer, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(T*& Pointer, T*& PPP, T*& OPP, T*& NPP, T*& POP, T*& OOP, T*& NOP, T*& PNP, T*& ONP, T*& NNP, T*& PPO, T*& OPO, T*& NPO, T*& POO, T*& NOO, T*& PNO, T*& ONO, T*& NNO, T*& PPN, T*& OPN, T*& NPN, T*& PON, T*& OON, T*& NON, T*& PNN, T*& ONN, T*& NNN) const {
    return FFront().GetConsecutives(Pointer, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(T*& Pointer, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) const {
    return FFront().GetConsecutives(Pointer, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::GetConsecutives(T*& Pointer, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) const {
    return FFront().GetConsecutives(Pointer, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::FillWith(T Value) {
    return FFront().FillWith(Value);
}
template <typename T>
__host__ __device__ std::pair<thrust::device_ptr<T>, size_t> BrendanCUDA::Fields::DField3<T>::Data() const {
    return FFront().Data();
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyAllIn(T* All, bool CopyFromHost) {
    return FFront().CopyAllIn(All, CopyFromHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyAllIn(T* All) {
    return FFront().CopyAllIn(All);
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyAllOut(T* All, bool CopyToHost) const {
    return FFront().CopyAllOut(All, CopyToHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyAllOut(T* All) const {
    return FFront().CopyAllOut(All);
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyValueIn(uint64_t Index, T* Value, bool CopyFromHost) {
    return FFront().CopyValueIn(Index, Value, CopyFromHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyValueIn(uint64_t Index, T* Value) {
    return FFront().CopyValueIn(Index, Value);
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyValueIn(uint32_3 Coordinates, T* Value, bool CopyFromHost) {
    return FFront().CopyValueIn(Coordinates, Value, CopyFromHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyValueIn(uint32_3 Coordinates, T* Value) {
    return FFront().CopyValueIn(Coordinates, Value);
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyFromHost) {
    return FFront().CopyValueIn(X, Y, Z, Value, CopyFromHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value) {
    return FFront().CopyValueIn(X, Y, Z, Value);
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyValueOut(uint64_t Index, T* Value, bool CopyToHost) const {
    return FFront().CopyValueOut(Index, Value, CopyToHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyValueOut(uint64_t Index, T* Value) const {
    return FFront().CopyValueOut(Index, Value);
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyValueOut(uint32_3 Coordinates, T* Value, bool CopyToHost) const {
    return FFront().CopyValueOut(Coordinates, Value, CopyToHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyValueOut(uint32_3 Coordinates, T* Value) const {
    return FFront().CopyValueOut(Coordinates, Value);
}
template <typename T>
__host__ void BrendanCUDA::Fields::DField3<T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyToHost) const {
    return FFront().CopyValueOut(X, Y, Z, Value, CopyToHost);
}
template <typename T>
__device__ void BrendanCUDA::Fields::DField3<T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value) const {
    return FFront().CopyValueOut(X, Y, Z, Value);
}
template <typename T>
__host__ T* BrendanCUDA::Fields::DField3<T>::GetAll(bool CopyToHost) const {
    return FFront().GetAll(CopyToHost);
}
template <typename T>
__device__ T* BrendanCUDA::Fields::DField3<T>::GetAll() const {
    return FFront().GetAll();
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::SetAll(T* All, bool CopyFromHost) {
    return FFront().SetAll(All, CopyFromHost);
}
template <typename T>
__host__ __device__ T BrendanCUDA::Fields::DField3<T>::GetValueAt(uint64_t Index) const {
    return FFront().GetValueAt(Index);
}
template <typename T>
__host__ __device__ T BrendanCUDA::Fields::DField3<T>::GetValueAt(uint32_3 Coordinates) const {
    return FFront().GetValueAt(Coordinates);
}
template <typename T>
__host__ __device__ T BrendanCUDA::Fields::DField3<T>::GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const {
    return FFront().GetValueAt(X, Y, Z);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::SetValueAt(uint64_t Index, T Value) {
    return FFront().SetValueAt(Index, Value);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::SetValueAt(uint32_3 Coordinates, T Value) {
    return FFront().SetValueAt(Coordinates, Value);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, T Value) {
    return FFront().SetValueAt(X, Y, Z, Value);
}