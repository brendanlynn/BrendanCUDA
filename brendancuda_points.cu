#include "brendancuda_points.h"
#include <cuda_runtime.h>
#include <cstdint>

using BrendanCUDA::uint32_2;
using BrendanCUDA::uint32_3;
using BrendanCUDA::uint32_4;
using BrendanCUDA::uint64_2;
using BrendanCUDA::uint64_3;
using BrendanCUDA::uint64_4;

__host__ __device__ uint32_t BrendanCUDA::Coordinates32_2ToIndex32_RM(uint32_2 Dimensions, uint32_2 Coordinates) {
    return Coordinates.y + Dimensions.y * (Coordinates.x);
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::Index32ToCoordinates32_2_RM(uint32_2 Dimensions, uint32_t Index) {
    uint32_2 r;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates32_2ToIndex32_CM(uint32_2 Dimensions, uint32_2 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y);
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::Index32ToCoordinates32_2_CM(uint32_2 Dimensions, uint32_t Index) {
    uint32_2 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates32_3ToIndex32_RM(uint32_3 Dimensions, uint32_3 Coordinates) {
    return Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x));
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Index32ToCoordinates32_3_RM(uint32_3 Dimensions, uint32_t Index) {
    uint32_3 r;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates32_3ToIndex32_CM(uint32_3 Dimensions, uint32_3 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z));
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Index32ToCoordinates32_3_CM(uint32_3 Dimensions, uint32_t Index) {
    uint32_3 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates32_4ToIndex32_RM(uint32_4 Dimensions, uint32_4 Coordinates) {
    return Coordinates.w + Dimensions.w * (Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x)));
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::Index32ToCoordinates32_4_RM(uint32_4 Dimensions, uint32_t Index) {
    uint32_4 r;
    r.w = Index % Dimensions.w;
    Index /= Dimensions.w;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates32_4ToIndex32_CM(uint32_4 Dimensions, uint32_4 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z + Dimensions.z * (Coordinates.w)));
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::Index32ToCoordinates32_4_CM(uint32_4 Dimensions, uint32_t Index) {
    uint32_4 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.w = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates32_2ToIndex64_RM(uint32_2 Dimensions, uint32_2 Coordinates) {
    return Coordinates.y + Dimensions.y * (Coordinates.x);
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::Index64ToCoordinates32_2_RM(uint32_2 Dimensions, uint64_t Index) {
    uint32_2 r;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates32_2ToIndex64_CM(uint32_2 Dimensions, uint32_2 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y);
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::Index64ToCoordinates32_2_CM(uint32_2 Dimensions, uint64_t Index) {
    uint32_2 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates32_3ToIndex64_RM(uint32_3 Dimensions, uint32_3 Coordinates) {
    return Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x));
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Index64ToCoordinates32_3_RM(uint32_3 Dimensions, uint64_t Index) {
    uint32_3 r;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates32_3ToIndex64_CM(uint32_3 Dimensions, uint32_3 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z));
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::Index64ToCoordinates32_3_CM(uint32_3 Dimensions, uint64_t Index) {
    uint32_3 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates32_4ToIndex64_RM(uint32_4 Dimensions, uint32_4 Coordinates) {
    return Coordinates.w + Dimensions.w * (Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x)));
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::Index64ToCoordinates32_4_RM(uint32_4 Dimensions, uint64_t Index) {
    uint32_4 r;
    r.w = Index % Dimensions.w;
    Index /= Dimensions.w;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates32_4ToIndex64_CM(uint32_4 Dimensions, uint32_4 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z + Dimensions.z * (Coordinates.w)));
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::Index64ToCoordinates32_4_CM(uint32_4 Dimensions, uint64_t Index) {
    uint32_4 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.w = Index;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates64_2ToIndex32_RM(uint64_2 Dimensions, uint64_2 Coordinates) {
    return Coordinates.y + Dimensions.y * (Coordinates.x);
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::Index32ToCoordinates64_2_RM(uint64_2 Dimensions, uint32_t Index) {
    uint64_2 r;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates64_2ToIndex32_CM(uint64_2 Dimensions, uint64_2 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y);
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::Index32ToCoordinates64_2_CM(uint64_2 Dimensions, uint32_t Index) {
    uint64_2 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates64_3ToIndex32_RM(uint64_3 Dimensions, uint64_3 Coordinates) {
    return Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x));
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::Index32ToCoordinates64_3_RM(uint64_3 Dimensions, uint32_t Index) {
    uint64_3 r;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates64_3ToIndex32_CM(uint64_3 Dimensions, uint64_3 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z));
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::Index32ToCoordinates64_3_CM(uint64_3 Dimensions, uint32_t Index) {
    uint64_3 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates64_4ToIndex32_RM(uint64_4 Dimensions, uint64_4 Coordinates) {
    return Coordinates.w + Dimensions.w * (Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x)));
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::Index32ToCoordinates64_4_RM(uint64_4 Dimensions, uint32_t Index) {
    uint64_4 r;
    r.w = Index % Dimensions.w;
    Index /= Dimensions.w;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint32_t BrendanCUDA::Coordinates64_4ToIndex32_CM(uint64_4 Dimensions, uint64_4 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z + Dimensions.z * (Coordinates.w)));
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::Index32ToCoordinates64_4_CM(uint64_4 Dimensions, uint32_t Index) {
    uint64_4 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.w = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates64_2ToIndex64_RM(uint64_2 Dimensions, uint64_2 Coordinates) {
    return Coordinates.y + Dimensions.y * (Coordinates.x);
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::Index64ToCoordinates64_2_RM(uint64_2 Dimensions, uint64_t Index) {
    uint64_2 r;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates64_2ToIndex64_CM(uint64_2 Dimensions, uint64_2 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y);
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::Index64ToCoordinates64_2_CM(uint64_2 Dimensions, uint64_t Index) {
    uint64_2 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates64_3ToIndex64_RM(uint64_3 Dimensions, uint64_3 Coordinates) {
    return Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x));
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::Index64ToCoordinates64_3_RM(uint64_3 Dimensions, uint64_t Index) {
    uint64_3 r;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates64_3ToIndex64_CM(uint64_3 Dimensions, uint64_3 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z));
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::Index64ToCoordinates64_3_CM(uint64_3 Dimensions, uint64_t Index) {
    uint64_3 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates64_4ToIndex64_RM(uint64_4 Dimensions, uint64_4 Coordinates) {
    return Coordinates.w + Dimensions.w * (Coordinates.z + Dimensions.z * (Coordinates.y + Dimensions.y * (Coordinates.x)));
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::Index64ToCoordinates64_4_RM(uint64_4 Dimensions, uint64_t Index) {
    uint64_4 r;
    r.w = Index % Dimensions.w;
    Index /= Dimensions.w;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.x = Index;
    return r;
}
__host__ __device__ uint64_t BrendanCUDA::Coordinates64_4ToIndex64_CM(uint64_4 Dimensions, uint64_4 Coordinates) {
    return Coordinates.x + Dimensions.x * (Coordinates.y + Dimensions.y * (Coordinates.z + Dimensions.z * (Coordinates.w)));
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::Index64ToCoordinates64_4_CM(uint64_4 Dimensions, uint64_t Index) {
    uint64_4 r;
    r.x = Index % Dimensions.x;
    Index /= Dimensions.x;
    r.y = Index % Dimensions.y;
    Index /= Dimensions.y;
    r.z = Index % Dimensions.z;
    Index /= Dimensions.z;
    r.w = Index;
    return r;
}
