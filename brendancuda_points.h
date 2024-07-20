#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_fixedvectors.h"

namespace BrendanCUDA {
    __host__ __device__ uint32_t Coordinates32_2ToIndex32_RM(uint32_2 Dimensions, uint32_2 Coordinates);
    __host__ __device__ uint32_2 Index32ToCoordinates32_2_RM(uint32_2 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates32_2ToIndex32_CM(uint32_2 Dimensions, uint32_2 Coordinates);
    __host__ __device__ uint32_2 Index32ToCoordinates32_2_CM(uint32_2 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates32_3ToIndex32_RM(uint32_3 Dimensions, uint32_3 Coordinates);
    __host__ __device__ uint32_3 Index32ToCoordinates32_3_RM(uint32_3 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates32_3ToIndex32_CM(uint32_3 Dimensions, uint32_3 Coordinates);
    __host__ __device__ uint32_3 Index32ToCoordinates32_3_CM(uint32_3 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates32_4ToIndex32_RM(uint32_4 Dimensions, uint32_4 Coordinates);
    __host__ __device__ uint32_4 Index32ToCoordinates32_4_RM(uint32_4 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates32_4ToIndex32_CM(uint32_4 Dimensions, uint32_4 Coordinates);
    __host__ __device__ uint32_4 Index32ToCoordinates32_4_CM(uint32_4 Dimensions, uint32_t Index);
    __host__ __device__ uint64_t Coordinates32_2ToIndex64_RM(uint32_2 Dimensions, uint32_2 Coordinates);
    __host__ __device__ uint32_2 Index64ToCoordinates32_2_RM(uint32_2 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates32_2ToIndex64_CM(uint32_2 Dimensions, uint32_2 Coordinates);
    __host__ __device__ uint32_2 Index64ToCoordinates32_2_CM(uint32_2 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates32_3ToIndex64_RM(uint32_3 Dimensions, uint32_3 Coordinates);
    __host__ __device__ uint32_3 Index64ToCoordinates32_3_RM(uint32_3 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates32_3ToIndex64_CM(uint32_3 Dimensions, uint32_3 Coordinates);
    __host__ __device__ uint32_3 Index64ToCoordinates32_3_CM(uint32_3 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates32_4ToIndex64_RM(uint32_4 Dimensions, uint32_4 Coordinates);
    __host__ __device__ uint32_4 Index64ToCoordinates32_4_RM(uint32_4 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates32_4ToIndex64_CM(uint32_4 Dimensions, uint32_4 Coordinates);
    __host__ __device__ uint32_4 Index64ToCoordinates32_4_CM(uint32_4 Dimensions, uint64_t Index);
    __host__ __device__ uint32_t Coordinates64_2ToIndex32_RM(uint64_2 Dimensions, uint64_2 Coordinates);
    __host__ __device__ uint64_2 Index32ToCoordinates64_2_RM(uint64_2 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates64_2ToIndex32_CM(uint64_2 Dimensions, uint64_2 Coordinates);
    __host__ __device__ uint64_2 Index32ToCoordinates64_2_CM(uint64_2 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates64_3ToIndex32_RM(uint64_3 Dimensions, uint64_3 Coordinates);
    __host__ __device__ uint64_3 Index32ToCoordinates64_3_RM(uint64_3 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates64_3ToIndex32_CM(uint64_3 Dimensions, uint64_3 Coordinates);
    __host__ __device__ uint64_3 Index32ToCoordinates64_3_CM(uint64_3 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates64_4ToIndex32_RM(uint64_4 Dimensions, uint64_4 Coordinates);
    __host__ __device__ uint64_4 Index32ToCoordinates64_4_RM(uint64_4 Dimensions, uint32_t Index);
    __host__ __device__ uint32_t Coordinates64_4ToIndex32_CM(uint64_4 Dimensions, uint64_4 Coordinates);
    __host__ __device__ uint64_4 Index32ToCoordinates64_4_CM(uint64_4 Dimensions, uint32_t Index);
    __host__ __device__ uint64_t Coordinates64_2ToIndex64_RM(uint64_2 Dimensions, uint64_2 Coordinates);
    __host__ __device__ uint64_2 Index64ToCoordinates64_2_RM(uint64_2 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates64_2ToIndex64_CM(uint64_2 Dimensions, uint64_2 Coordinates);
    __host__ __device__ uint64_2 Index64ToCoordinates64_2_CM(uint64_2 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates64_3ToIndex64_RM(uint64_3 Dimensions, uint64_3 Coordinates);
    __host__ __device__ uint64_3 Index64ToCoordinates64_3_RM(uint64_3 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates64_3ToIndex64_CM(uint64_3 Dimensions, uint64_3 Coordinates);
    __host__ __device__ uint64_3 Index64ToCoordinates64_3_CM(uint64_3 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates64_4ToIndex64_RM(uint64_4 Dimensions, uint64_4 Coordinates);
    __host__ __device__ uint64_4 Index64ToCoordinates64_4_RM(uint64_4 Dimensions, uint64_t Index);
    __host__ __device__ uint64_t Coordinates64_4ToIndex64_CM(uint64_4 Dimensions, uint64_4 Coordinates);
    __host__ __device__ uint64_4 Index64ToCoordinates64_4_CM(uint64_4 Dimensions, uint64_t Index);
}