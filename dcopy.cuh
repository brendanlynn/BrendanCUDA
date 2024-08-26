#pragma once

#include <cstdint>
#include <cuda_runtime.h>

__device__ void deviceMemcpy(void* Destination, const void* Source, size_t Count);