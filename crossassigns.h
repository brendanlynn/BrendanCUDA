#pragma once

#include "errorhelp.h"
#include <cuda_runtime.h>

namespace bcuda {
    template <typename _T>
    static inline _T GetVR(_T* DevicePointer) {
        _T v;
        ThrowIfBad(cudaMemcpy(&v, DevicePointer, sizeof(_T), cudaMemcpyDeviceToHost));
        return v;
    }
    template <typename _T>
    static inline void SetVR(_T* DevicePointer, _T Value) {
        ThrowIfBad(cudaMemcpy(DevicePointer, &Value, sizeof(_T), cudaMemcpyHostToDevice));
    }
}