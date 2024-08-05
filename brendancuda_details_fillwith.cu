#include "brendancuda_details_fillwith.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <device_launch_parameters.h>

__global__ void fillWithKernel(void* Array, void* Value, size_t ValueSize) {
    memcpy((uint8_t*)Array + blockIdx.x * ValueSize, Value, ValueSize);
}

__host__ __device__ __forceinline void BrendanCUDA::details::FillWith(void* Array, size_t ArrayElementCount, void* Value, size_t ValueSize) {
    fillWithKernel<<<ArrayElementCount, 1>>>(Array, Value, ValueSize);
}