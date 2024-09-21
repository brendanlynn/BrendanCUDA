#pragma once

#include "errorhelp.h"
#include <cuda_runtime.h>

namespace bcuda {
    template <typename _T>
    _T GetVR(_T* DevicePointer);
    template <typename _T>
    void SetVR(_T* DevicePointer, _T Value);
}

template <typename _T>
_T bcuda::GetVR(_T* DevicePointer) {
    _T v;
    ThrowIfBad(cudaMemcpy(&v, DevicePointer, sizeof(_T), cudaMemcpyDeviceToHost));
    return v;
}

template <typename _T>
void bcuda::SetVR(_T* DevicePointer, _T Value) {
    ThrowIfBad(cudaMemcpy(DevicePointer, &Value, sizeof(_T), cudaMemcpyHostToDevice));
}