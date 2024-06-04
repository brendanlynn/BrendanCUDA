#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

#include "brendancuda_macros.cuh"
#include "brendancuda_devicecopy.cuh"

namespace BrendanCUDA {
    namespace Fields {
        template <typename T>
        class Field3 final {
        public:
            __host__ __device__ Field3(uint3 Dimensions);
            __host__ __device__ Field3(dim3 Dimensions);
            __host__ __device__ Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ);

            __device__ Field3(uint3 Dimensions, T* All);
            __device__ Field3(dim3 Dimensions, T* All);
            __device__ Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All);

            __host__ Field3(uint3 Dimensions, T* All, bool CopyFromHost);
            __host__ Field3(dim3 Dimensions, T* All, bool CopyFromHost);
            __host__ Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All, bool CopyFromHost);

            __host__ __device__ uint32_t LengthX() const;
            __host__ __device__ uint32_t LengthY() const;
            __host__ __device__ uint32_t LengthZ() const;

            __host__ __device__ uint3 Dimensions() const;
            __host__ __device__ dim3 DimensionsD() const;

            __host__ __device__ size_t SizeOnGPU() const;

            __host__ void CopyAllIn(T* All, bool CopyFromHost);
            __device__ void CopyAllIn(T* All);
            __host__ void CopyAllOut(T* All, bool CopyToHost) const;
            __device__ void CopyAllOut(T* All) const;
            __host__ void CopyValueIn(size_t Index, T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(size_t Index, T* Value);
            __host__ void CopyValueIn(uint3 Coordinates, T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint3 Coordinates, T* Value);
            __host__ void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyFromHost);
            __device__ void CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value);
            __host__ void CopyValueOut(size_t Index, T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(size_t Index, T* Value) const;
            __host__ void CopyValueOut(uint3 Coordinates, T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint3 Coordinates, T* Value) const;
            __host__ void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyToHost) const;
            __device__ void CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value) const;

            __host__ T* GetAll(bool CopyToHost) const;
            __device__ T* GetAll() const;
            __host__ __device__ void SetAll(T* All, bool CopyFromHost);

            __host__ __device__ T GetValueAt(size_t Index) const;
            __host__ __device__ T GetValueAt(uint3 Coordinates) const;
            __host__ __device__ T GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const;

            __host__ __device__ void SetValueAt(size_t Index, T Value);
            __host__ __device__ void SetValueAt(uint3 Coordinates, T Value);
            __host__ __device__ void SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, T Value);

            __host__ __device__ void Dispose();

            __host__ __device__ size_t CoordinatesToIndex(uint3 Coordinates) const;
            __host__ __device__ size_t CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ uint3 IndexToCoordinates(size_t Index) const;

            __host__ __device__ T* IndexToPointer(size_t Index) const;
            __host__ __device__ size_t PointerToIndex(T* Pointer) const;

            __host__ __device__ T* CoordinatesToPointer(uint3 Coordinates) const;
            __host__ __device__ T* CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const;
            __host__ __device__ uint3 PointerToCoordinates(T* Pointer) const;

            __host__ __device__ void GetConsecutives(size_t Index, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const;
            __host__ __device__ void GetConsecutives(size_t Index, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const;

            __host__ __device__ void GetConsecutives(size_t Index, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const;
            __host__ __device__ void GetConsecutives(size_t Index, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const;

            __host__ __device__ void GetConsecutives(size_t Index, uint3* POO, uint3* NOO, uint3* OPO, uint3* ONO, uint3* OOP, uint3* OON) const;
            __host__ __device__ void GetConsecutives(size_t Index, uint3* PPP, uint3* OPP, uint3* NPP, uint3* POP, uint3* OOP, uint3* NOP, uint3* PNP, uint3* ONP, uint3* NNP, uint3* PPO, uint3* OPO, uint3* NPO, uint3* POO, uint3* NOO, uint3* PNO, uint3* ONO, uint3* NNO, uint3* PPN, uint3* OPN, uint3* NPN, uint3* PON, uint3* OON, uint3* NON, uint3* PNN, uint3* ONN, uint3* NNN) const;

            __host__ __device__ void GetConsecutives(uint3 Coordinates, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const;
            __host__ __device__ void GetConsecutives(uint3 Coordinates, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const;

            __host__ __device__ void GetConsecutives(uint3 Coordinates, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const;
            __host__ __device__ void GetConsecutives(uint3 Coordinates, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const;

            __host__ __device__ void GetConsecutives(uint3 Coordinates, uint3* POO, uint3* NOO, uint3* OPO, uint3* ONO, uint3* OOP, uint3* OON) const;
            __host__ __device__ void GetConsecutives(uint3 Coordinates, uint3* PPP, uint3* OPP, uint3* NPP, uint3* POP, uint3* OOP, uint3* NOP, uint3* PNP, uint3* ONP, uint3* NNP, uint3* PPO, uint3* OPO, uint3* NPO, uint3* POO, uint3* NOO, uint3* PNO, uint3* ONO, uint3* NNO, uint3* PPN, uint3* OPN, uint3* NPN, uint3* PON, uint3* OON, uint3* NON, uint3* PNN, uint3* ONN, uint3* NNN) const;

            __host__ __device__ void GetConsecutives(T** Pointer, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const;
            __host__ __device__ void GetConsecutives(T** Pointer, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const;

            __host__ __device__ void GetConsecutives(T** Pointer, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const;
            __host__ __device__ void GetConsecutives(T** Pointer, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const;

            __host__ __device__ void GetConsecutives(T** Pointer, uint3* POO, uint3* NOO, uint3* OPO, uint3* ONO, uint3* OOP, uint3* OON) const;
            __host__ __device__ void GetConsecutives(T** Pointer, uint3* PPP, uint3* OPP, uint3* NPP, uint3* POP, uint3* OOP, uint3* NOP, uint3* PNP, uint3* ONP, uint3* NNP, uint3* PPO, uint3* OPO, uint3* NPO, uint3* POO, uint3* NOO, uint3* PNO, uint3* ONO, uint3* NNO, uint3* PPN, uint3* OPN, uint3* NPN, uint3* PON, uint3* OON, uint3* NON, uint3* PNN, uint3* ONN, uint3* NNN) const;

            __host__ __device__ void FillWith(T Value);

            __host__ __device__ std::pair<T*, size_t> Data() const;
        private:
            uint32_t lengthX;
            uint32_t lengthY;
            uint32_t lengthZ;

            T* cudaArray;

            __host__ __device__ void GetConsecutives(uint3 Coordinates, size_t Index, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const;
            __host__ __device__ void GetConsecutives(uint3 Coordinates, size_t Index, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const;

            __host__ __device__ void GetConsecutives(uint3 Coordinates, size_t Index, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const;
            __host__ __device__ void GetConsecutives(uint3 Coordinates, size_t Index, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const;

            __host__ __device__ void GetIndexDeltas(uint3 Coordinates, size_t Index, int64_t* DXP, int64_t* DXN, int64_t* DYP, int64_t* DYN, int64_t* DZP, int64_t* DZN) const;

            __host__ __device__ void GetNewCoordinates(uint3 Coordinates, uint32_t* XP, uint32_t* XN, uint32_t* YP, uint32_t* YN, uint32_t* ZP, uint32_t* ZN) const;
        };
    }
}

template <typename T>
__global__ void FillWithKernel(BrendanCUDA::Fields::Field3<T> This, T Value) {
    *This.CoordinatesToPointer(make_int3(blockIdx.x, blockIdx.y, blockIdx.z)) = Value;
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::FillWith(T Value) {
    FillWithKernel<T><<<DimensionsD(), 1>>>(this, Value);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, size_t Index, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    GetIndexDeltas(Coordinates, Index, &dXP, &dXN, &dYP, &dYN, &dZP, &dZN);
    *POO = Index + dXP;
    *NOO = Index + dXN;
    *OPO = Index + dYP;
    *ONO = Index + dYN;
    *OOP = Index + dZP;
    *OON = Index + dZN;
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, size_t Index, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    GetIndexDeltas(Coordinates, Index, &dXP, &dXN, &dYP, &dYN, &dZP, &dZN);
    *PPP = Index + dXP + dYP + dZP;
    *OPP = Index + 000 + dYP + dZP;
    *NPP = Index + dXN + dYP + dZP;
    *POP = Index + dXP + 000 + dZP;
    *OOP = Index + 000 + 000 + dZP;
    *NOP = Index + dXN + 000 + dZP;
    *PNP = Index + dXP + dYN + dZP;
    *ONP = Index + 000 + dYN + dZP;
    *NNP = Index + dXN + dYN + dZP;
    *PPO = Index + dXP + dYP + 000;
    *OPO = Index + 000 + dYP + 000;
    *NPO = Index + dXN + dYP + 000;
    *POO = Index + dXP + 000 + 000;
    *OOO = Index + 000 + 000 + 000;
    *NOO = Index + dXN + 000 + 000;
    *PNO = Index + dXP + dYN + 000;
    *ONO = Index + 000 + dYN + 000;
    *NNO = Index + dXN + dYN + 000;
    *PPN = Index + dXP + dYP + dZN;
    *OPN = Index + 000 + dYP + dZN;
    *NPN = Index + dXN + dYP + dZN;
    *PON = Index + dXP + 000 + dZN;
    *OON = Index + 000 + 000 + dZN;
    *NON = Index + dXN + 000 + dZN;
    *PNN = Index + dXP + dYN + dZN;
    *ONN = Index + 000 + dYN + dZN;
    *NNN = Index + dXN + dYN + dZN;
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, size_t Index, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const {
    size_t iPOO;
    size_t iNOO;
    size_t iOPO;
    size_t iONO;
    size_t iOOP;
    size_t iOON;
    GetConsecutives(Coordinates, Index, &iPOO, &iNOO, &iOPO, &iONO, &iOOP, &iOON);
    *POO = &cudaArray[iPOO];
    *NOO = &cudaArray[iNOO];
    *OPO = &cudaArray[iOPO];
    *ONO = &cudaArray[iONO];
    *OOP = &cudaArray[iOOP];
    *OON = &cudaArray[iOON];
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, size_t Index, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const {
    size_t iPPP;
    size_t iOPP;
    size_t iNPP;
    size_t iPOP;
    size_t iOOP;
    size_t iNOP;
    size_t iPNP;
    size_t iONP;
    size_t iNNP;
    size_t iPPO;
    size_t iOPO;
    size_t iNPO;
    size_t iPOO;
    size_t iNOO;
    size_t iPNO;
    size_t iONO;
    size_t iNNO;
    size_t iPPN;
    size_t iOPN;
    size_t iNPN;
    size_t iPON;
    size_t iOON;
    size_t iNON;
    size_t iPNN;
    size_t iONN;
    size_t iNNN;
    GetConsecutives(Coordinates, Index, &iPPP, &iOPP, &iNPP, &iPOP, &iOOP, &iNOP, &iPNP, &iONP, &iNNP, &iPPO, &iOPO, &iNPO, &iPOO, &iNOO, &iPNO, &iONO, &iNNO, &iPPN, &iOPN, &iNPN, &iPON, &iOON, &iNON, &iPNN, &iONN, &iNNN);
    *PPP = &cudaArray[iPPP];
    *OPP = &cudaArray[iOPP];
    *NPP = &cudaArray[iNPP];
    *POP = &cudaArray[iPOP];
    *OOP = &cudaArray[iOOP];
    *NOP = &cudaArray[iNOP];
    *PNP = &cudaArray[iPNP];
    *ONP = &cudaArray[iONP];
    *NNP = &cudaArray[iNNP];
    *PPO = &cudaArray[iPPO];
    *OPO = &cudaArray[iOPO];
    *NPO = &cudaArray[iNPO];
    *POO = &cudaArray[iPOO];
    *OOO = &cudaArray[iOOO];
    *NOO = &cudaArray[iNOO];
    *PNO = &cudaArray[iPNO];
    *ONO = &cudaArray[iONO];
    *NNO = &cudaArray[iNNO];
    *PPN = &cudaArray[iPPN];
    *OPN = &cudaArray[iOPN];
    *NPN = &cudaArray[iNPN];
    *PON = &cudaArray[iPON];
    *OON = &cudaArray[iOON];
    *NON = &cudaArray[iNON];
    *PNN = &cudaArray[iPNN];
    *ONN = &cudaArray[iONN];
    *NNN = &cudaArray[iNNN];
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetIndexDeltas(uint3 Coordinates, size_t Index, int64_t* DXP, int64_t* DXN, int64_t* DYP, int64_t* DYN, int64_t* DZP, int64_t* DZN) const {
    int64_t dX = lengthY * lengthZ;
    if (Coordinates.x == 0) {
        *DXP = dX;
        *DXN = ((int64_t)lengthX - (int64_t)Coordinates.x - 1i64) * dX;
    }
    else if (Coordinates.x == lengthX - 1i64) {
        *DXP = -(int64_t)Coordinates.x * dX;
        *DXN = -dX;
    }
    else {
        *DXP = dX;
        *DXN = -dX;
    }

    if (Coordinates.y == 0) {
        *DYP = (int64_t)lengthZ;
        *DYN = ((int64_t)lengthY - (int64_t)Coordinates.y - 1i64) * (int64_t)lengthZ;
    }
    else if (Coordinates.y == lengthY - 1) {
        *DYP = -(int64_t)Coordinates.y * (int64_t)lengthZ;
        *DYN = -(int64_t)lengthZ;
    }
    else {
        *DYP = (int64_t)lengthZ;
        *DYN = -(int64_t)lengthZ;
    }

    if (Coordinates.z == 0) {
        *DZP = 1i64;
        *DZN = (int64_t)lengthZ - (int64_t)Coordinates.z - 1i64;
    }
    else if (Coordinates.z == lengthZ - 1) {
        *DZP = -(int64_t)Coordinates.z;
        *DZN = -1i64;
    }
    else {
        *DZP = 1i64;
        *DZN = -1i64;
    }
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetNewCoordinates(uint3 Coordinates, uint32_t* XP, uint32_t* XN, uint32_t* YP, uint32_t* YN, uint32_t* ZP, uint32_t* ZN) const {
    if (Coordinates.x == 0) {
        *XP = 1i64;
        *XN = (int64_t)lengthX - 1i64;
    }
    else if (Coordinates.x == lengthX - 1) {
        *XP = 0i64;
        *XN = (int64_t)lengthX - 2i64;
    }
    else {
        *XP = (int64_t)Coordinates.x + 1i64;
        *XN = (int64_t)Coordinates.x - 1i64;
    }

    if (Coordinates.y == 0) {
        *YP = 1i64;
        *YN = (int64_t)lengthY - 1i64;
    }
    else if (Coordinates.y == lengthY - 1) {
        *YP = 0i64;
        *YN = (int64_t)lengthY - 2i64;
    }
    else {
        *YP = (int64_t)Coordinates.y + 1i64;
        *YN = (int64_t)Coordinates.y - 1i64;
    }

    if (Coordinates.z == 0) {
        *ZP = 1i64;
        *ZN = (int64_t)lengthZ - 1i64;
    }
    else if (Coordinates.z == lengthZ - 1) {
        *ZP = 0i64;
        *ZN = (int64_t)lengthZ - 2i64;
    }
    else {
        *ZP = (int64_t)Coordinates.z + 1i64;
        *ZN = (int64_t)Coordinates.z - 1i64;
    }
}
template <typename T>
__host__ __device__ BrendanCUDA::Fields::Field3<T>::Field3(uint3 Dimensions)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z) { }
template <typename T>
__host__ __device__ BrendanCUDA::Fields::Field3<T>::Field3(dim3 Dimensions)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z) { }
template <typename T>
__host__ __device__ BrendanCUDA::Fields::Field3<T>::Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ) {
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
#if IS_ON_DEVICE
        cudaArray = new T[LengthX * LengthY * LengthZ];
#else
        cudaError_t e = cudaMalloc(&cudaArray, SizeOnGPU());
        if (e) {
            throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
        }
#endif
    }
}
template <typename T>
__device__ BrendanCUDA::Fields::Field3<T>::Field3(uint3 Dimensions, T* All)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename T>
__device__ BrendanCUDA::Fields::Field3<T>::Field3(dim3 Dimensions, T* All)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename T>
__device__ BrendanCUDA::Fields::Field3<T>::Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All)
    : Field3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All);
}
template <typename T>
__host__ BrendanCUDA::Fields::Field3<T>::Field3(uint3 Dimensions, T* All, bool CopyFromHost)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename T>
__host__ BrendanCUDA::Fields::Field3<T>::Field3(dim3 Dimensions, T* All, bool CopyFromHost)
    : Field3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename T>
__host__ BrendanCUDA::Fields::Field3<T>::Field3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All, bool CopyFromHost)
    : Field3(LengthX, LengthY, LengthZ) {
    CopyAllIn(All, CopyFromHost);
}
template <typename T>
__host__ __device__ uint32_t BrendanCUDA::Fields::Field3<T>::LengthX() const {
    return lengthX;
}
template <typename T>
__host__ __device__ uint32_t BrendanCUDA::Fields::Field3<T>::LengthY() const {
    return lengthY;
}
template <typename T>
__host__ __device__ uint32_t BrendanCUDA::Fields::Field3<T>::LengthZ() const {
    return lengthZ;
}
template <typename T>
__host__ __device__ uint3 BrendanCUDA::Fields::Field3<T>::Dimensions() const {
    return make_uint3(lengthX, lengthY, lengthZ);
}
template <typename T>
__host__ __device__ dim3 BrendanCUDA::Fields::Field3<T>::DimensionsD() const {
    return dim3(lengthX, lengthY, lengthZ);
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::Fields::Field3<T>::SizeOnGPU() const {
    return (((size_t)lengthX) * ((size_t)lengthY) * ((size_t)lengthZ)) * sizeof(T);
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyAllIn(T* All, bool CopyFromHost) {
    cudaError_t e = cudaMemcpy(cudaArray, All, SizeOnGPU(), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyAllIn(T* All) {
    deviceMemcpy(cudaArray, All, SizeOnGPU());
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyAllOut(T* All, bool CopyToHost) const {
    cudaError_t e = cudaMemcpy(All, cudaArray, SizeOnGPU(), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyAllOut(T* All) const {
    deviceMemcpy(All, cudaArray, SizeOnGPU());
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyValueIn(size_t Index, T* Value, bool CopyFromHost) {
    cudaError_t e = cudaMemcpy(IndexToPointer(Index), Value, sizeof(T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyValueIn(size_t Index, T* Value) {
    deviceMemcpy(IndexToPointer(Index), Value, sizeof(T));
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyValueIn(uint3 Coordinates, T* Value, bool CopyFromHost) {
    cudaError_t e = cudaMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyValueIn(uint3 Coordinates, T* Value) {
    deviceMemcpy(CoordinatesToPointer(Coordinates), Value, sizeof(T));
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyFromHost) {
    cudaError_t e = cudaMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(T), CopyFromHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyValueIn(uint32_t X, uint32_t Y, uint32_t Z, T* Value) {
    deviceMemcpy(CoordinatesToPointer(X, Y, Z), Value, sizeof(T));
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyValueOut(size_t Index, T* Value, bool CopyToHost) const {
    cudaError_t e = cudaMemcpy(Value, IndexToPointer(Index), sizeof(T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyValueOut(size_t Index, T* Value) const {
    deviceMemcpy(Value, IndexToPointer(Index), sizeof(T));
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyValueOut(uint3 Coordinates, T* Value, bool CopyToHost) const {
    cudaError_t e = cudaMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyValueOut(uint3 Coordinates, T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(Coordinates), sizeof(T));
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value, bool CopyToHost) const {
#if _DEBUG
    auto t = cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(T), CopyToHost ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
        throw runtime_error("A CUDA error occured. Error #" + to_string(e) + ".");
}
template <typename T>
__device__ void BrendanCUDA::Fields::Field3<T>::CopyValueOut(uint32_t X, uint32_t Y, uint32_t Z, T* Value) const {
    deviceMemcpy(Value, CoordinatesToPointer(X, Y, Z), sizeof(T));
}
template <typename T>
__host__ T* BrendanCUDA::Fields::Field3<T>::GetAll(bool CopyToHost) const {
    T* a;
    if (CopyToHost) {
        a = new T[lengthX * lengthY * lengthZ * lengthD];
    }
    else {
        cudaMalloc(&a, SizeOnGPU());
    }
    CopyAllOut(a, CopyToHost);
    return a;
}
template <typename T>
__device__ T* BrendanCUDA::Fields::Field3<T>::GetAll() const {
    T* a = new T[lengthX * lengthY * lengthZ * lengthD];
    CopyAllOut(a, false);
    return a;
}
template <typename T>
__host__ void BrendanCUDA::Fields::Field3<T>::SetAll(T* All, bool CopyFromHost) {
    CopyAllIn(All, CopyFromHost);
}
template <typename T>
__host__ __device__ T BrendanCUDA::Fields::Field3<T>::GetValueAt(size_t Index) const {
    T v;
#if IS_ON_HOST
    CopyValueOut(Index, &v, true);
#else
    CopyValueOut(Index, &v);
#endif
    return v;
}
template <typename T>
__host__ __device__ T BrendanCUDA::Fields::Field3<T>::GetValueAt(uint3 Coordinates) const {
    return GetValueAt(Coordinates.x, Coordinates.y, Coordinates.z);
}
template <typename T>
__host__ __device__ T BrendanCUDA::Fields::Field3<T>::GetValueAt(uint32_t X, uint32_t Y, uint32_t Z) const {
    T v;
#if IS_ON_HOST
    CopyValueOut(X, Y, Z, &v, true);
#else
    CopyValueOut(X, Y, Z, &v);
#endif
    return v;
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::SetValueAt(size_t Index, T Value) {
#if IS_ON_HOST
    CopyValueIn(Index, &Value, true);
#else
    CopyValueIn(Index, &Value);
#endif
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::SetValueAt(uint3 Coordinates, T Value) {
    SetValueAt(Coordinates.x, Coordinates.y, Coordinates.z, Value);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::SetValueAt(uint32_t X, uint32_t Y, uint32_t Z, T Value) {
#if IS_ON_HOST
    CopyValueIn(X, Y, Z, &Value, true);
#else
    CopyValueIn(X, Y, Z, &Value);
#endif
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::Dispose() {
#if IS_ON_HOST
    cudaFree(cudaArray);
#else
    delete[] cudaArray;
#endif
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::Fields::Field3<T>::CoordinatesToIndex(uint3 Coordinates) const {
    return CoordinatesToIndex(Coordinates.x, Coordinates.y, Coordinates.z);
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::Fields::Field3<T>::CoordinatesToIndex(uint32_t X, uint32_t Y, uint32_t Z) const {
    return (X * lengthY + Y) * lengthZ + Z;
}
template <typename T>
__host__ __device__ uint3 BrendanCUDA::Fields::Field3<T>::IndexToCoordinates(size_t Index) const {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    z = Index % lengthZ;
    Index /= lengthZ;
    y = Index % lengthY;
    Index /= lengthY;
    x = Index % lengthX;
    return make_uint3(x, y, z);
}
template <typename T>
__host__ __device__ T* BrendanCUDA::Fields::Field3<T>::IndexToPointer(size_t Index) const {
    return &cudaArray[Index];
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::Fields::Field3<T>::PointerToIndex(T* Pointer) const {
    return ((uint64_t)(reinterpret_cast<uintptr_t>(Pointer) - reinterpret_cast<uintptr_t>(cudaArray))) / (uint64_t)sizeof(T);
}
template <typename T>
__host__ __device__ T* BrendanCUDA::Fields::Field3<T>::CoordinatesToPointer(uint3 Coordinates) const {
    return CoordinatesToPointer(Coordinates.x, Coordinates.y, Coordinates.z);
}
template <typename T>
__host__ __device__ T* BrendanCUDA::Fields::Field3<T>::CoordinatesToPointer(uint32_t X, uint32_t Y, uint32_t Z) const {
    return IndexToPointer(CoordinatesToIndex(X, Y, Z));
}
template <typename T>
__host__ __device__ uint3 BrendanCUDA::Fields::Field3<T>::PointerToCoordinates(T* Pointer) const {
    return IndexToCoordinates(PointerToIndex(Pointer));
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(size_t Index, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const {
    GetConsecutives(IndexToCoordinates(Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(size_t Index, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const {
    GetConsecutives(IndexToCoordinates(Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(size_t Index, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const {
    GetConsecutives(IndexToCoordinates(Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(size_t Index, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const {
    GetConsecutives(IndexToCoordinates(Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(size_t Index, uint3* POO, uint3* NOO, uint3* OPO, uint3* ONO, uint3* OOP, uint3* OON) const {
    GetConsecutives(IndexToCoordinates(Index), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(size_t Index, uint3* PPP, uint3* OPP, uint3* NPP, uint3* POP, uint3* OOP, uint3* NOP, uint3* PNP, uint3* ONP, uint3* NNP, uint3* PPO, uint3* OPO, uint3* NPO, uint3* POO, uint3* NOO, uint3* PNO, uint3* ONO, uint3* NNO, uint3* PPN, uint3* OPN, uint3* NPN, uint3* PON, uint3* OON, uint3* NON, uint3* PNN, uint3* ONN, uint3* NNN) const {
    GetConsecutives(IndexToCoordinates(Index), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const {
    GetConsecutives(Coordinates, CoordinatesToIndex(Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const {
    GetConsecutives(Coordinates, CoordinatesToIndex(Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const {
    GetConsecutives(Coordinates, CoordinatesToIndex(Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const {
    GetConsecutives(Coordinates, CoordinatesToIndex(Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, uint3* POO, uint3* NOO, uint3* OPO, uint3* ONO, uint3* OOP, uint3* OON) const {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;
    uint32_t zP;
    uint32_t zN;

    GetNewCoordinates(Coordinates, &xP, &xN, &yP, &yN, &zP, &zN);

    *POO = make_uint3(xP, Coordinates.y, Coordinates.z);
    *NOO = make_uint3(xN, Coordinates.y, Coordinates.z);
    *OPO = make_uint3(Coordinates.x, yP, Coordinates.z);
    *ONO = make_uint3(Coordinates.x, yN, Coordinates.z);
    *OOP = make_uint3(Coordinates.x, Coordinates.y, zP);
    *OON = make_uint3(Coordinates.x, Coordinates.y, zN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(uint3 Coordinates, uint3* PPP, uint3* OPP, uint3* NPP, uint3* POP, uint3* OOP, uint3* NOP, uint3* PNP, uint3* ONP, uint3* NNP, uint3* PPO, uint3* OPO, uint3* NPO, uint3* POO, uint3* NOO, uint3* PNO, uint3* ONO, uint3* NNO, uint3* PPN, uint3* OPN, uint3* NPN, uint3* PON, uint3* OON, uint3* NON, uint3* PNN, uint3* ONN, uint3* NNN) const {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;
    uint32_t zP;
    uint32_t zN;

    GetNewCoordinates(Coordinates, &xP, &xN, &yP, &yN, &zP, &zN);

    *PPP = make_uint3(xP, yP, zP);
    *OPP = make_uint3(Coordinates.x, yP, zP);
    *NPP = make_uint3(xN, yP, zP);
    *POP = make_uint3(xP, Coordinates.y, zP);
    *OOP = make_uint3(Coordinates.x, Coordinates.y, zP);
    *NOP = make_uint3(xN, Coordinates.y, zP);
    *PNP = make_uint3(xP, yN, zP);
    *ONP = make_uint3(Coordinates.x, yN, zP);
    *NNP = make_uint3(xN, yN, zP);
    *PPO = make_uint3(xP, yP, Coordinates.z);
    *OPO = make_uint3(Coordinates.x, yP, Coordinates.z);
    *NPO = make_uint3(xN, yP, Coordinates.z);
    *POO = make_uint3(xP, Coordinates.y, Coordinates.z);
    *NOO = make_uint3(xN, Coordinates.y, Coordinates.z);
    *PNO = make_uint3(xP, yN, Coordinates.z);
    *ONO = make_uint3(Coordinates.x, yN, Coordinates.z);
    *NNO = make_uint3(xN, yN, Coordinates.z);
    *PPP = make_uint3(xP, yP, zN);
    *OPP = make_uint3(Coordinates.x, yP, zN);
    *NPP = make_uint3(xN, yP, zN);
    *POP = make_uint3(xP, Coordinates.y, zN);
    *OOP = make_uint3(Coordinates.x, Coordinates.y, zN);
    *NOP = make_uint3(xN, Coordinates.y, zN);
    *PNP = make_uint3(xP, yN, zN);
    *ONP = make_uint3(Coordinates.x, yN, zN);
    *NNP = make_uint3(xN, yN, zN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(T** Pointer, size_t* POO, size_t* NOO, size_t* OPO, size_t* ONO, size_t* OOP, size_t* OON) const {
    GetConsecutives(PointerToIndex(Pointer), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(T** Pointer, size_t* PPP, size_t* OPP, size_t* NPP, size_t* POP, size_t* OOP, size_t* NOP, size_t* PNP, size_t* ONP, size_t* NNP, size_t* PPO, size_t* OPO, size_t* NPO, size_t* POO, size_t* NOO, size_t* PNO, size_t* ONO, size_t* NNO, size_t* PPN, size_t* OPN, size_t* NPN, size_t* PON, size_t* OON, size_t* NON, size_t* PNN, size_t* ONN, size_t* NNN) const {
    GetConsecutives(PointerToIndex(Pointer), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(T** Pointer, T** POO, T** NOO, T** OPO, T** ONO, T** OOP, T** OON) const {
    GetConsecutives(PointerToIndex(Pointer), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(T** Pointer, T** PPP, T** OPP, T** NPP, T** POP, T** OOP, T** NOP, T** PNP, T** ONP, T** NNP, T** PPO, T** OPO, T** NPO, T** POO, T** NOO, T** PNO, T** ONO, T** NNO, T** PPN, T** OPN, T** NPN, T** PON, T** OON, T** NON, T** PNN, T** ONN, T** NNN) const {
    GetConsecutives(PointerToIndex(Pointer), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(T** Pointer, uint3* POO, uint3* NOO, uint3* OPO, uint3* ONO, uint3* OOP, uint3* OON) const {
    GetConsecutives(PointerToCoordinates(Pointer), POO, NOO, OPO, ONO, OOP, OON);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::Field3<T>::GetConsecutives(T** Pointer, uint3* PPP, uint3* OPP, uint3* NPP, uint3* POP, uint3* OOP, uint3* NOP, uint3* PNP, uint3* ONP, uint3* NNP, uint3* PPO, uint3* OPO, uint3* NPO, uint3* POO, uint3* NOO, uint3* PNO, uint3* ONO, uint3* NNO, uint3* PPN, uint3* OPN, uint3* NPN, uint3* PON, uint3* OON, uint3* NON, uint3* PNN, uint3* ONN, uint3* NNN) const {
    GetConsecutives(PointerToCoordinates(Pointer), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
template <typename T>
__host__ __device__ std::pair<T*, size_t> BrendanCUDA::Fields::Field3<T>::Data() const {
    return { cudaArray, lengthX * lengthY * lengthZ };
}