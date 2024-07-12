#pragma once

#include <cuda_runtime.h>
#include <cstdint>

__device__ void deviceMemcpy(void* Destination, const void* Source, size_t Count);