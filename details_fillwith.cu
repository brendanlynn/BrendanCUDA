#include "details_fillwith.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fillWithKernel(void* Array, void* Value, size_t ValueSize) {
    memcpy((uint8_t*)Array + blockIdx.x * ValueSize, Value, ValueSize);
}

__forceinline void brendancuda::details::FillWith(void* Array, size_t ArrayElementCount, void* Value, size_t ValueSize) {
    fillWithKernel<<<ArrayElementCount, 1>>>(Array, Value, ValueSize);
}