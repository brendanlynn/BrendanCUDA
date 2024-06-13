#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace BrendanCUDA {
    void ThrowIfBad(cudaError_t e);
    void ThrowIfBad(cublasStatus_t e);
}