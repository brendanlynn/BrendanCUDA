#include "brendancuda_cudaerrorhelpers.h"

#include <exception>
#include <string>

void BrendanCUDA::ThrowIfBad(cudaError_t e) {
    if (e) {
        throw std::exception(("A CUDA error occured: cudaError_t #" + std::to_string(e) + ".").c_str());
    }
}

void BrendanCUDA::ThrowIfBad(cublasStatus_t e) {
    if (e) {
        throw std::exception(("A CUDA error occured: cublasStatus_t #" + std::to_string(e) + ".").c_str());
    }
}