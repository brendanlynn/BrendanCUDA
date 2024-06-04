#pragma once

#include "brendancuda_fields_field3.cuh"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <typename T>
        class DField3 final {
        public:
            __host__ __device__ DField3(uint3 Dimensions);
            __host__ __device__ DField3(dim3 Dimensions);
            __host__ __device__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ);

            __device__ DField3(uint3 Dimensions, T* All);
            __device__ DField3(dim3 Dimensions, T* All);
            __device__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All);

            __host__ DField3(uint3 Dimensions, T* All, bool CopyFromHost);
            __host__ DField3(dim3 Dimensions, T* All, bool CopyFromHost);
            __host__ DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All, bool CopyFromHost);

            __host__ __device__ uint32_t LengthX() const;
            __host__ __device__ uint32_t LengthY() const;
            __host__ __device__ uint32_t LengthZ() const;

            __host__ __device__ uint3 Dimensions() const;
            __host__ __device__ dim3 DimensionsD() const;

            __host__ __device__ size_t SizeOnGPU() const;

            __host__ __device__ void Dispose();

            __host__ __device__ Field3<T> FFront() const;
            __host__ __device__ Field3<T> FBack() const;
            __host__ __device__ void Reverse();
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
__host__ __device__ BrendanCUDA::Fields::DField3<T>::DField3(uint3 Dimensions)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z) { }
template <typename T>
__host__ __device__ BrendanCUDA::Fields::DField3<T>::DField3(dim3 Dimensions)
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
#if IS_ON_DEVICE
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
__device__ BrendanCUDA::Fields::DField3<T>::DField3(uint3 Dimensions, T* All)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All) { }
template <typename T>
__device__ BrendanCUDA::Fields::DField3<T>::DField3(dim3 Dimensions, T* All)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All) { }
template <typename T>
__device__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All)
    : DField3(LengthX, LengthY, LengthZ) {
    FFront().CopyAllIn(All);
}
template <typename T>
__host__ BrendanCUDA::Fields::DField3<T>::DField3(uint3 Dimensions, T* All, bool CopyFromHost)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename T>
__host__ BrendanCUDA::Fields::DField3<T>::DField3(dim3 Dimensions, T* All, bool CopyFromHost)
    : DField3(Dimensions.x, Dimensions.y, Dimensions.z, All, CopyFromHost) { }
template <typename T>
__host__ BrendanCUDA::Fields::DField3<T>::DField3(uint32_t LengthX, uint32_t LengthY, uint32_t LengthZ, T* All, bool CopyFromHost)
    : DField3(LengthX, LengthY, LengthZ) {
    FFront().CopyAllIn(All, CopyFromHost);
}
template <typename T>
__host__ __device__ void BrendanCUDA::Fields::DField3<T>::Dispose() {
#if IS_ON_HOST
    cudaFree(cudaArrayF);
    cudaFree(cudaArrayB);
#else
    delete[] cudaArrayF;
    delete[] cudaArrayB;
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
__host__ __device__ uint3 BrendanCUDA::Fields::DField3<T>::Dimensions() const {
    return make_uint3(lengthX, lengthY, lengthZ);
}
template <typename T>
__host__ __device__ dim3 BrendanCUDA::Fields::DField3<T>::DimensionsD() const {
    return dim3(lengthX, lengthY, lengthZ);
}
template <typename T>
__host__ __device__ size_t BrendanCUDA::Fields::DField3<T>::SizeOnGPU() const {
    return ((((size_t)lengthX) * ((size_t)lengthY) * ((size_t)lengthZ)) * sizeof(T)) << 1;
}