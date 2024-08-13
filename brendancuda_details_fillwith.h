#pragma once

#include "brendancuda_errorhelp.h"
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace details {
        void FillWith(void* Array, size_t ArrayElementCount, void* Value, size_t ValueSize);
    }
    template <typename _T>
    void FillWith(_T* Array, size_t Length, _T Value) {
        void* cValue;
        ThrowIfBad(cudaMalloc(&cValue, sizeof(_T)));
        ThrowIfBad(cudaMemcpy(cValue, &Value, sizeof(_T), cudaMemcpyHostToDevice));
        details::FillWith(Array, Length, cValue, sizeof(_T));
        ThrowIfBad(cudaFree(cValue));
    }
}