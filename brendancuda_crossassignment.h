#pragma once

#include <cuda_runtime.h>
#include "brendancuda_cudaerrorhelpers.h"

namespace BrendanCUDA {
    template <typename T>
    T GetVR(T* DevicePointer);
    template <typename T>
    void SetVR(T* DevicePointer, T Value);
}

template <typename T>
T BrendanCUDA::GetVR(T* DevicePointer) {
    T v;
    ThrowIfBad(cudaMemcpy(&v, DevicePointer, sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

template <typename T>
void BrendanCUDA::SetVR(T* DevicePointer, T Value) {
    ThrowIfBad(cudaMemcpy(DevicePointer, &Value, sizeof(T), cudaMemcpyHostToDevice));
}