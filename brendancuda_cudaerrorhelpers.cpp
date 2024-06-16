#include "brendancuda_cudaerrorhelpers.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <exception>
#include <string>

template <>
void BrendanCUDA::ThrowIfBad<cudaError_t>(cudaError_t e) {
    if (e) {
        throw std::exception(("A CUDA error occured: cudaError_t #" + std::to_string(e) + ".").c_str());
    }
}

template <>
void BrendanCUDA::ThrowIfBad<cublasStatus_t>(cublasStatus_t e) {
    if (e) {
        throw std::exception(("A CUDA error occured: cublasStatus_t #" + std::to_string(e) + ".").c_str());
    }
}

template <>
void BrendanCUDA::ThrowIfBad<CUresult>(CUresult e) {
    if (e) {
        throw std::exception(("A CUDA error occured: CUresult #" + std::to_string(e) + ".").c_str());
    }
}

template <typename T>
void BrendanCUDA::ThrowIfBad(T e) {
    if (e) {
        throw std::exception(("An error occured: error #" + std::to_string(e) + ".").c_str());
    }
}

template void BrendanCUDA::ThrowIfBad<cudaError_t>(cudaError_t e);
template void BrendanCUDA::ThrowIfBad<cublasStatus_t>(cublasStatus_t e);
template void BrendanCUDA::ThrowIfBad<CUresult>(CUresult e);
template void BrendanCUDA::ThrowIfBad<int8_t>(int8_t e);
template void BrendanCUDA::ThrowIfBad<uint8_t>(uint8_t e);
template void BrendanCUDA::ThrowIfBad<int16_t>(int16_t e);
template void BrendanCUDA::ThrowIfBad<uint16_t>(uint16_t e);
template void BrendanCUDA::ThrowIfBad<int32_t>(int32_t e);
template void BrendanCUDA::ThrowIfBad<uint32_t>(uint32_t e);
template void BrendanCUDA::ThrowIfBad<int64_t>(int64_t e);
template void BrendanCUDA::ThrowIfBad<uint64_t>(uint64_t e);