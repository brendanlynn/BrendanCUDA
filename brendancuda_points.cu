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

__host__ __device__ void getIndexDeltas2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN) {
    if (Coordinates.x == 0) {
        DXP = (int32_t)Dimensions.y;
        DXN = ((int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32) * (int32_t)Dimensions.y;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int32_t)Coordinates.x * (int32_t)Dimensions.y;
        DXN = -(int32_t)Dimensions.y;
    }
    else {
        DXP = (int32_t)Dimensions.y;
        DXN = -(int32_t)Dimensions.y;
    }

    if (Coordinates.y == 0) {
        DYP = 1i32;
        DYN = (int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y;
        DYN = -1i32;
    }
    else {
        DYP = 1i32;
        DYN = -1i32;
    }
}
__host__ __device__ void getIndexDeltas3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN, int32_t& DZP, int32_t& DZN) {
    int32_t dX = Dimensions.y * Dimensions.z;
    if (Coordinates.x == 0) {
        DXP = dX;
        DXN = ((int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32) * dX;
    }
    else if (Coordinates.x == Dimensions.x - 1i32) {
        DXP = -(int32_t)Coordinates.x * dX;
        DXN = -dX;
    }
    else {
        DXP = dX;
        DXN = -dX;
    }

    if (Coordinates.y == 0) {
        DYP = (int32_t)Dimensions.z;
        DYN = ((int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32) * (int32_t)Dimensions.z;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y * (int32_t)Dimensions.z;
        DYN = -(int32_t)Dimensions.z;
    }
    else {
        DYP = (int32_t)Dimensions.z;
        DYN = -(int32_t)Dimensions.z;
    }

    if (Coordinates.z == 0) {
        DZP = 1i32;
        DZN = (int32_t)Dimensions.z - (int32_t)Coordinates.z - 1i32;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        DZP = -(int32_t)Coordinates.z;
        DZN = -1i32;
    }
    else {
        DZP = 1i32;
        DZN = -1i32;
    }
}
__host__ __device__ void getIndexDeltas2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN) {
    if (Coordinates.y == 0) {
        DYP = (int32_t)Dimensions.x;
        DYN = ((int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32) * (int32_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y * (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }
    else {
        DYP = (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i32;
        DXN = (int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int32_t)Coordinates.x;
        DXN = -1i32;
    }
    else {
        DXP = 1i32;
        DXN = -1i32;
    }
}
__host__ __device__ void getIndexDeltas3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN, int32_t& DZP, int32_t& DZN) {
    int32_t dZ = Dimensions.y * Dimensions.x;
    if (Coordinates.z == 0) {
        DZP = dZ;
        DZN = ((int32_t)Dimensions.z - (int32_t)Coordinates.z - 1i32) * dZ;
    }
    else if (Coordinates.z == Dimensions.z - 1i32) {
        DZP = -(int32_t)Coordinates.z * dZ;
        DZN = -dZ;
    }
    else {
        DZP = dZ;
        DZN = -dZ;
    }

    if (Coordinates.y == 0) {
        DYP = (int32_t)Dimensions.x;
        DYN = ((int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32) * (int32_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y * (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }
    else {
        DYP = (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i32;
        DXN = (int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int32_t)Coordinates.x;
        DXN = -1i32;
    }
    else {
        DXP = 1i32;
        DXN = -1i32;
    }
}
__host__ __device__ void getConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getNewCoordinates2(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t& XP, uint32_t& XN, uint32_t& YP, uint32_t& YN) {
    if (!Coordinates.x) {
        XP = 1i32;
        XN = (int32_t)Dimensions.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i32;
        XN = (int32_t)Dimensions.x - 2i32;
    }
    else {
        XP = (int32_t)Coordinates.x + 1i32;
        XN = (int32_t)Coordinates.x - 1i32;
    }

    if (!Coordinates.y) {
        YP = 1i32;
        YN = (int32_t)Dimensions.y - 1i32;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i32;
        YN = (int32_t)Dimensions.y - 2i32;
    }
    else {
        YP = (int32_t)Coordinates.y + 1i32;
        YN = (int32_t)Coordinates.y - 1i32;
    }
}
__host__ __device__ void getNewCoordinates3(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t& XP, uint32_t& XN, uint32_t& YP, uint32_t& YN, uint32_t& ZP, uint32_t& ZN) {
    if (!Coordinates.x) {
        XP = 1i32;
        XN = (int32_t)Dimensions.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i32;
        XN = (int32_t)Dimensions.x - 2i32;
    }
    else {
        XP = (int32_t)Coordinates.x + 1i32;
        XN = (int32_t)Coordinates.x - 1i32;
    }

    if (!Coordinates.y) {
        YP = 1i32;
        YN = (int32_t)Dimensions.y - 1i32;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i32;
        YN = (int32_t)Dimensions.y - 2i32;
    }
    else {
        YP = (int32_t)Coordinates.y + 1i32;
        YN = (int32_t)Coordinates.y - 1i32;
    }

    if (!Coordinates.z) {
        ZP = 1i32;
        ZN = (int32_t)Dimensions.z - 1i32;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        ZP = 0i32;
        ZN = (int32_t)Dimensions.z - 2i32;
    }
    else {
        ZP = (int32_t)Coordinates.z + 1i32;
        ZN = (int32_t)Coordinates.z - 1i32;
    }
}
__host__ __device__ void getIndexDeltas2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN) {
    if (Coordinates.x == 0) {
        DXP = (int32_t)Dimensions.y;
        DXN = ((int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32) * (int32_t)Dimensions.y;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int32_t)Coordinates.x * (int32_t)Dimensions.y;
        DXN = -(int32_t)Dimensions.y;
    }
    else {
        DXP = (int32_t)Dimensions.y;
        DXN = -(int32_t)Dimensions.y;
    }

    if (Coordinates.y == 0) {
        DYP = 1i32;
        DYN = (int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y;
        DYN = -1i32;
    }
    else {
        DYP = 1i32;
        DYN = -1i32;
    }
}
__host__ __device__ void getIndexDeltas3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN, int32_t& DZP, int32_t& DZN) {
    int32_t dX = Dimensions.y * Dimensions.z;
    if (Coordinates.x == 0) {
        DXP = dX;
        DXN = ((int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32) * dX;
    }
    else if (Coordinates.x == Dimensions.x - 1i32) {
        DXP = -(int32_t)Coordinates.x * dX;
        DXN = -dX;
    }
    else {
        DXP = dX;
        DXN = -dX;
    }

    if (Coordinates.y == 0) {
        DYP = (int32_t)Dimensions.z;
        DYN = ((int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32) * (int32_t)Dimensions.z;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y * (int32_t)Dimensions.z;
        DYN = -(int32_t)Dimensions.z;
    }
    else {
        DYP = (int32_t)Dimensions.z;
        DYN = -(int32_t)Dimensions.z;
    }

    if (Coordinates.z == 0) {
        DZP = 1i32;
        DZN = (int32_t)Dimensions.z - (int32_t)Coordinates.z - 1i32;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        DZP = -(int32_t)Coordinates.z;
        DZN = -1i32;
    }
    else {
        DZP = 1i32;
        DZN = -1i32;
    }
}
__host__ __device__ void getIndexDeltas2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN) {
    if (Coordinates.y == 0) {
        DYP = (int32_t)Dimensions.x;
        DYN = ((int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32) * (int32_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y * (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }
    else {
        DYP = (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i32;
        DXN = (int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int32_t)Coordinates.x;
        DXN = -1i32;
    }
    else {
        DXP = 1i32;
        DXN = -1i32;
    }
}
__host__ __device__ void getIndexDeltas3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t Index, int32_t& DXP, int32_t& DXN, int32_t& DYP, int32_t& DYN, int32_t& DZP, int32_t& DZN) {
    int32_t dZ = Dimensions.y * Dimensions.x;
    if (Coordinates.z == 0) {
        DZP = dZ;
        DZN = ((int32_t)Dimensions.z - (int32_t)Coordinates.z - 1i32) * dZ;
    }
    else if (Coordinates.z == Dimensions.z - 1i32) {
        DZP = -(int32_t)Coordinates.z * dZ;
        DZN = -dZ;
    }
    else {
        DZP = dZ;
        DZN = -dZ;
    }

    if (Coordinates.y == 0) {
        DYP = (int32_t)Dimensions.x;
        DYN = ((int32_t)Dimensions.y - (int32_t)Coordinates.y - 1i32) * (int32_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int32_t)Coordinates.y * (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }
    else {
        DYP = (int32_t)Dimensions.x;
        DYN = -(int32_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i32;
        DXN = (int32_t)Dimensions.x - (int32_t)Coordinates.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int32_t)Coordinates.x;
        DXN = -1i32;
    }
    else {
        DXP = 1i32;
        DXN = -1i32;
    }
}
__host__ __device__ void getConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    int32_t dXP;
    int32_t dXN;
    int32_t dYP;
    int32_t dYN;
    int32_t dZP;
    int32_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getNewCoordinates2(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t& XP, uint32_t& XN, uint32_t& YP, uint32_t& YN) {
    if (!Coordinates.x) {
        XP = 1i32;
        XN = (int32_t)Dimensions.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i32;
        XN = (int32_t)Dimensions.x - 2i32;
    }
    else {
        XP = (int32_t)Coordinates.x + 1i32;
        XN = (int32_t)Coordinates.x - 1i32;
    }

    if (!Coordinates.y) {
        YP = 1i32;
        YN = (int32_t)Dimensions.y - 1i32;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i32;
        YN = (int32_t)Dimensions.y - 2i32;
    }
    else {
        YP = (int32_t)Coordinates.y + 1i32;
        YN = (int32_t)Coordinates.y - 1i32;
    }
}
__host__ __device__ void getNewCoordinates3(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t& XP, uint32_t& XN, uint32_t& YP, uint32_t& YN, uint32_t& ZP, uint32_t& ZN) {
    if (!Coordinates.x) {
        XP = 1i32;
        XN = (int32_t)Dimensions.x - 1i32;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i32;
        XN = (int32_t)Dimensions.x - 2i32;
    }
    else {
        XP = (int32_t)Coordinates.x + 1i32;
        XN = (int32_t)Coordinates.x - 1i32;
    }

    if (!Coordinates.y) {
        YP = 1i32;
        YN = (int32_t)Dimensions.y - 1i32;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i32;
        YN = (int32_t)Dimensions.y - 2i32;
    }
    else {
        YP = (int32_t)Coordinates.y + 1i32;
        YN = (int32_t)Coordinates.y - 1i32;
    }

    if (!Coordinates.z) {
        ZP = 1i32;
        ZN = (int32_t)Dimensions.z - 1i32;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        ZP = 0i32;
        ZN = (int32_t)Dimensions.z - 2i32;
    }
    else {
        ZP = (int32_t)Coordinates.z + 1i32;
        ZN = (int32_t)Coordinates.z - 1i32;
    }
}
__host__ __device__ void getIndexDeltas2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN) {
    if (Coordinates.x == 0) {
        DXP = (int64_t)Dimensions.y;
        DXN = ((int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64) * (int64_t)Dimensions.y;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int64_t)Coordinates.x * (int64_t)Dimensions.y;
        DXN = -(int64_t)Dimensions.y;
    }
    else {
        DXP = (int64_t)Dimensions.y;
        DXN = -(int64_t)Dimensions.y;
    }

    if (Coordinates.y == 0) {
        DYP = 1i64;
        DYN = (int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y;
        DYN = -1i64;
    }
    else {
        DYP = 1i64;
        DYN = -1i64;
    }
}
__host__ __device__ void getIndexDeltas3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN, int64_t& DZP, int64_t& DZN) {
    int64_t dX = Dimensions.y * Dimensions.z;
    if (Coordinates.x == 0) {
        DXP = dX;
        DXN = ((int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64) * dX;
    }
    else if (Coordinates.x == Dimensions.x - 1i64) {
        DXP = -(int64_t)Coordinates.x * dX;
        DXN = -dX;
    }
    else {
        DXP = dX;
        DXN = -dX;
    }

    if (Coordinates.y == 0) {
        DYP = (int64_t)Dimensions.z;
        DYN = ((int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64) * (int64_t)Dimensions.z;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y * (int64_t)Dimensions.z;
        DYN = -(int64_t)Dimensions.z;
    }
    else {
        DYP = (int64_t)Dimensions.z;
        DYN = -(int64_t)Dimensions.z;
    }

    if (Coordinates.z == 0) {
        DZP = 1i64;
        DZN = (int64_t)Dimensions.z - (int64_t)Coordinates.z - 1i64;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        DZP = -(int64_t)Coordinates.z;
        DZN = -1i64;
    }
    else {
        DZP = 1i64;
        DZN = -1i64;
    }
}
__host__ __device__ void getIndexDeltas2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN) {
    if (Coordinates.y == 0) {
        DYP = (int64_t)Dimensions.x;
        DYN = ((int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64) * (int64_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y * (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }
    else {
        DYP = (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i64;
        DXN = (int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int64_t)Coordinates.x;
        DXN = -1i64;
    }
    else {
        DXP = 1i64;
        DXN = -1i64;
    }
}
__host__ __device__ void getIndexDeltas3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN, int64_t& DZP, int64_t& DZN) {
    int64_t dZ = Dimensions.y * Dimensions.x;
    if (Coordinates.z == 0) {
        DZP = dZ;
        DZN = ((int64_t)Dimensions.z - (int64_t)Coordinates.z - 1i64) * dZ;
    }
    else if (Coordinates.z == Dimensions.z - 1i64) {
        DZP = -(int64_t)Coordinates.z * dZ;
        DZN = -dZ;
    }
    else {
        DZP = dZ;
        DZN = -dZ;
    }

    if (Coordinates.y == 0) {
        DYP = (int64_t)Dimensions.x;
        DYN = ((int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64) * (int64_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y * (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }
    else {
        DYP = (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i64;
        DXN = (int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int64_t)Coordinates.x;
        DXN = -1i64;
    }
    else {
        DXP = 1i64;
        DXN = -1i64;
    }
}
__host__ __device__ void getConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getNewCoordinates2(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t& XP, uint64_t& XN, uint64_t& YP, uint64_t& YN) {
    if (!Coordinates.x) {
        XP = 1i64;
        XN = (int64_t)Dimensions.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i64;
        XN = (int64_t)Dimensions.x - 2i64;
    }
    else {
        XP = (int64_t)Coordinates.x + 1i64;
        XN = (int64_t)Coordinates.x - 1i64;
    }

    if (!Coordinates.y) {
        YP = 1i64;
        YN = (int64_t)Dimensions.y - 1i64;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i64;
        YN = (int64_t)Dimensions.y - 2i64;
    }
    else {
        YP = (int64_t)Coordinates.y + 1i64;
        YN = (int64_t)Coordinates.y - 1i64;
    }
}
__host__ __device__ void getNewCoordinates3(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t& XP, uint64_t& XN, uint64_t& YP, uint64_t& YN, uint64_t& ZP, uint64_t& ZN) {
    if (!Coordinates.x) {
        XP = 1i64;
        XN = (int64_t)Dimensions.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i64;
        XN = (int64_t)Dimensions.x - 2i64;
    }
    else {
        XP = (int64_t)Coordinates.x + 1i64;
        XN = (int64_t)Coordinates.x - 1i64;
    }

    if (!Coordinates.y) {
        YP = 1i64;
        YN = (int64_t)Dimensions.y - 1i64;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i64;
        YN = (int64_t)Dimensions.y - 2i64;
    }
    else {
        YP = (int64_t)Coordinates.y + 1i64;
        YN = (int64_t)Coordinates.y - 1i64;
    }

    if (!Coordinates.z) {
        ZP = 1i64;
        ZN = (int64_t)Dimensions.z - 1i64;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        ZP = 0i64;
        ZN = (int64_t)Dimensions.z - 2i64;
    }
    else {
        ZP = (int64_t)Coordinates.z + 1i64;
        ZN = (int64_t)Coordinates.z - 1i64;
    }
}
__host__ __device__ void getIndexDeltas2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN) {
    if (Coordinates.x == 0) {
        DXP = (int64_t)Dimensions.y;
        DXN = ((int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64) * (int64_t)Dimensions.y;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int64_t)Coordinates.x * (int64_t)Dimensions.y;
        DXN = -(int64_t)Dimensions.y;
    }
    else {
        DXP = (int64_t)Dimensions.y;
        DXN = -(int64_t)Dimensions.y;
    }

    if (Coordinates.y == 0) {
        DYP = 1i64;
        DYN = (int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y;
        DYN = -1i64;
    }
    else {
        DYP = 1i64;
        DYN = -1i64;
    }
}
__host__ __device__ void getIndexDeltas3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN, int64_t& DZP, int64_t& DZN) {
    int64_t dX = Dimensions.y * Dimensions.z;
    if (Coordinates.x == 0) {
        DXP = dX;
        DXN = ((int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64) * dX;
    }
    else if (Coordinates.x == Dimensions.x - 1i64) {
        DXP = -(int64_t)Coordinates.x * dX;
        DXN = -dX;
    }
    else {
        DXP = dX;
        DXN = -dX;
    }

    if (Coordinates.y == 0) {
        DYP = (int64_t)Dimensions.z;
        DYN = ((int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64) * (int64_t)Dimensions.z;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y * (int64_t)Dimensions.z;
        DYN = -(int64_t)Dimensions.z;
    }
    else {
        DYP = (int64_t)Dimensions.z;
        DYN = -(int64_t)Dimensions.z;
    }

    if (Coordinates.z == 0) {
        DZP = 1i64;
        DZN = (int64_t)Dimensions.z - (int64_t)Coordinates.z - 1i64;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        DZP = -(int64_t)Coordinates.z;
        DZN = -1i64;
    }
    else {
        DZP = 1i64;
        DZN = -1i64;
    }
}
__host__ __device__ void getIndexDeltas2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN) {
    if (Coordinates.y == 0) {
        DYP = (int64_t)Dimensions.x;
        DYN = ((int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64) * (int64_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y * (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }
    else {
        DYP = (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i64;
        DXN = (int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int64_t)Coordinates.x;
        DXN = -1i64;
    }
    else {
        DXP = 1i64;
        DXN = -1i64;
    }
}
__host__ __device__ void getIndexDeltas3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t Index, int64_t& DXP, int64_t& DXN, int64_t& DYP, int64_t& DYN, int64_t& DZP, int64_t& DZN) {
    int64_t dZ = Dimensions.y * Dimensions.x;
    if (Coordinates.z == 0) {
        DZP = dZ;
        DZN = ((int64_t)Dimensions.z - (int64_t)Coordinates.z - 1i64) * dZ;
    }
    else if (Coordinates.z == Dimensions.z - 1i64) {
        DZP = -(int64_t)Coordinates.z * dZ;
        DZN = -dZ;
    }
    else {
        DZP = dZ;
        DZN = -dZ;
    }

    if (Coordinates.y == 0) {
        DYP = (int64_t)Dimensions.x;
        DYN = ((int64_t)Dimensions.y - (int64_t)Coordinates.y - 1i64) * (int64_t)Dimensions.x;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        DYP = -(int64_t)Coordinates.y * (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }
    else {
        DYP = (int64_t)Dimensions.x;
        DYN = -(int64_t)Dimensions.x;
    }

    if (Coordinates.x == 0) {
        DXP = 1i64;
        DXN = (int64_t)Dimensions.x - (int64_t)Coordinates.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        DXP = -(int64_t)Coordinates.x;
        DXN = -1i64;
    }
    else {
        DXP = 1i64;
        DXN = -1i64;
    }
}
__host__ __device__ void getConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PO = Index + dXP;
    NO = Index + dXN;
    OP = Index + dYP;
    ON = Index + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    POO = Index + dXP;
    NOO = Index + dXN;
    OPO = Index + dYP;
    ONO = Index + dYN;
    OOP = Index + dZP;
    OON = Index + dZN;
}
__host__ __device__ void getConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_RM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    getIndexDeltas2_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN);
    PP = Index + dXP + dYP;
    OP = Index + 000 + dYP;
    NP = Index + dXN + dYP;
    PO = Index + dXP + 000;
    NO = Index + dXN + 000;
    PN = Index + dXP + dYN;
    ON = Index + 000 + dYN;
    NN = Index + dXN + dYN;
}
__host__ __device__ void getConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    int64_t dXP;
    int64_t dXN;
    int64_t dYP;
    int64_t dYN;
    int64_t dZP;
    int64_t dZN;
    getIndexDeltas3_CM(Dimensions, Coordinates, Index, dXP, dXN, dYP, dYN, dZP, dZN);
    PPP = Index + dXP + dYP + dZP;
    OPP = Index + 000 + dYP + dZP;
    NPP = Index + dXN + dYP + dZP;
    POP = Index + dXP + 000 + dZP;
    OOP = Index + 000 + 000 + dZP;
    NOP = Index + dXN + 000 + dZP;
    PNP = Index + dXP + dYN + dZP;
    ONP = Index + 000 + dYN + dZP;
    NNP = Index + dXN + dYN + dZP;
    PPO = Index + dXP + dYP + 000;
    OPO = Index + 000 + dYP + 000;
    NPO = Index + dXN + dYP + 000;
    POO = Index + dXP + 000 + 000;
    NOO = Index + dXN + 000 + 000;
    PNO = Index + dXP + dYN + 000;
    ONO = Index + 000 + dYN + 000;
    NNO = Index + dXN + dYN + 000;
    PPN = Index + dXP + dYP + dZN;
    OPN = Index + 000 + dYP + dZN;
    NPN = Index + dXN + dYP + dZN;
    PON = Index + dXP + 000 + dZN;
    OON = Index + 000 + 000 + dZN;
    NON = Index + dXN + 000 + dZN;
    PNN = Index + dXP + dYN + dZN;
    ONN = Index + 000 + dYN + dZN;
    NNN = Index + dXN + dYN + dZN;
}
__host__ __device__ void getNewCoordinates2(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t& XP, uint64_t& XN, uint64_t& YP, uint64_t& YN) {
    if (!Coordinates.x) {
        XP = 1i64;
        XN = (int64_t)Dimensions.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i64;
        XN = (int64_t)Dimensions.x - 2i64;
    }
    else {
        XP = (int64_t)Coordinates.x + 1i64;
        XN = (int64_t)Coordinates.x - 1i64;
    }

    if (!Coordinates.y) {
        YP = 1i64;
        YN = (int64_t)Dimensions.y - 1i64;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i64;
        YN = (int64_t)Dimensions.y - 2i64;
    }
    else {
        YP = (int64_t)Coordinates.y + 1i64;
        YN = (int64_t)Coordinates.y - 1i64;
    }
}
__host__ __device__ void getNewCoordinates3(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t& XP, uint64_t& XN, uint64_t& YP, uint64_t& YN, uint64_t& ZP, uint64_t& ZN) {
    if (!Coordinates.x) {
        XP = 1i64;
        XN = (int64_t)Dimensions.x - 1i64;
    }
    else if (Coordinates.x == Dimensions.x - 1) {
        XP = 0i64;
        XN = (int64_t)Dimensions.x - 2i64;
    }
    else {
        XP = (int64_t)Coordinates.x + 1i64;
        XN = (int64_t)Coordinates.x - 1i64;
    }

    if (!Coordinates.y) {
        YP = 1i64;
        YN = (int64_t)Dimensions.y - 1i64;
    }
    else if (Coordinates.y == Dimensions.y - 1) {
        YP = 0i64;
        YN = (int64_t)Dimensions.y - 2i64;
    }
    else {
        YP = (int64_t)Coordinates.y + 1i64;
        YN = (int64_t)Coordinates.y - 1i64;
    }

    if (!Coordinates.z) {
        ZP = 1i64;
        ZN = (int64_t)Dimensions.z - 1i64;
    }
    else if (Coordinates.z == Dimensions.z - 1) {
        ZP = 0i64;
        ZN = (int64_t)Dimensions.z - 2i64;
    }
    else {
        ZP = (int64_t)Coordinates.z + 1i64;
        ZN = (int64_t)Coordinates.z - 1i64;
    }
}


__host__ __device__ void getConsecutives2(uint32_2 Dimensions, uint32_2 Coordinates, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;

    getNewCoordinates2(Dimensions, Coordinates, xP, xN, yP, yN);

    PO = uint32_2(xP, Coordinates.y);
    NO = uint32_2(xN, Coordinates.y);
    OP = uint32_2(Coordinates.x, yP);
    ON = uint32_2(Coordinates.x, yN);
}
__host__ __device__ void getConsecutives3(uint32_3 Dimensions, uint32_3 Coordinates, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;
    uint32_t zP;
    uint32_t zN;

    getNewCoordinates3(Dimensions, Coordinates, xP, xN, yP, yN, zP, zN);

    POO = uint32_3(xP, Coordinates.y, Coordinates.z);
    NOO = uint32_3(xN, Coordinates.y, Coordinates.z);
    OPO = uint32_3(Coordinates.x, yP, Coordinates.z);
    ONO = uint32_3(Coordinates.x, yN, Coordinates.z);
    OOP = uint32_3(Coordinates.x, Coordinates.y, zP);
    OON = uint32_3(Coordinates.x, Coordinates.y, zN);
}
__host__ __device__ void getConsecutives2(uint32_2 Dimensions, uint32_2 Coordinates, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;

    getNewCoordinates2(Dimensions, Coordinates, xP, xN, yP, yN);

    PP = uint32_2(xP, yP);
    OP = uint32_2(Coordinates.x, yP);
    NP = uint32_2(xN, yP);
    PO = uint32_2(xP, Coordinates.y);
    NO = uint32_2(xN, Coordinates.y);
    PN = uint32_2(xP, yN);
    ON = uint32_2(Coordinates.x, yN);
    NN = uint32_2(xN, yN);
}
__host__ __device__ void getConsecutives3(uint32_3 Dimensions, uint32_3 Coordinates, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;
    uint32_t zP;
    uint32_t zN;

    getNewCoordinates3(Dimensions, Coordinates, xP, xN, yP, yN, zP, zN);

    PPP = uint32_3(xP, yP, zP);
    OPP = uint32_3(Coordinates.x, yP, zP);
    NPP = uint32_3(xN, yP, zP);
    POP = uint32_3(xP, Coordinates.y, zP);
    OOP = uint32_3(Coordinates.x, Coordinates.y, zP);
    NOP = uint32_3(xN, Coordinates.y, zP);
    PNP = uint32_3(xP, yN, zP);
    ONP = uint32_3(Coordinates.x, yN, zP);
    NNP = uint32_3(xN, yN, zP);
    PPO = uint32_3(xP, yP, Coordinates.z);
    OPO = uint32_3(Coordinates.x, yP, Coordinates.z);
    NPO = uint32_3(xN, yP, Coordinates.z);
    POO = uint32_3(xP, Coordinates.y, Coordinates.z);
    NOO = uint32_3(xN, Coordinates.y, Coordinates.z);
    PNO = uint32_3(xP, yN, Coordinates.z);
    ONO = uint32_3(Coordinates.x, yN, Coordinates.z);
    NNO = uint32_3(xN, yN, Coordinates.z);
    PPN = uint32_3(xP, yP, zN);
    OPN = uint32_3(Coordinates.x, yP, zN);
    NPN = uint32_3(xN, yP, zN);
    PON = uint32_3(xP, Coordinates.y, zN);
    OON = uint32_3(Coordinates.x, Coordinates.y, zN);
    NON = uint32_3(xN, Coordinates.y, zN);
    PNN = uint32_3(xP, yN, zN);
    ONN = uint32_3(Coordinates.x, yN, zN);
    NNN = uint32_3(xN, yN, zN);
}
__host__ __device__ void getConsecutives2(uint64_2 Dimensions, uint64_2 Coordinates, uint64_2& PO, uint64_2& NO, uint64_2& OP, uint64_2& ON) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;

    getNewCoordinates2(Dimensions, Coordinates, xP, xN, yP, yN);

    PO = uint64_2(xP, Coordinates.y);
    NO = uint64_2(xN, Coordinates.y);
    OP = uint64_2(Coordinates.x, yP);
    ON = uint64_2(Coordinates.x, yN);
}
__host__ __device__ void getConsecutives3(uint64_3 Dimensions, uint64_3 Coordinates, uint64_3& POO, uint64_3& NOO, uint64_3& OPO, uint64_3& ONO, uint64_3& OOP, uint64_3& OON) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;
    uint32_t zP;
    uint32_t zN;

    getNewCoordinates3(Dimensions, Coordinates, xP, xN, yP, yN, zP, zN);

    POO = uint64_3(xP, Coordinates.y, Coordinates.z);
    NOO = uint64_3(xN, Coordinates.y, Coordinates.z);
    OPO = uint64_3(Coordinates.x, yP, Coordinates.z);
    ONO = uint64_3(Coordinates.x, yN, Coordinates.z);
    OOP = uint64_3(Coordinates.x, Coordinates.y, zP);
    OON = uint64_3(Coordinates.x, Coordinates.y, zN);
}
__host__ __device__ void getConsecutives2(uint64_2 Dimensions, uint64_2 Coordinates, uint64_2& PP, uint64_2& OP, uint64_2& NP, uint64_2& PO, uint64_2& NO, uint64_2& PN, uint64_2& ON, uint64_2& NN) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;

    getNewCoordinates2(Dimensions, Coordinates, xP, xN, yP, yN);

    PP = uint64_2(xP, yP);
    OP = uint64_2(Coordinates.x, yP);
    NP = uint64_2(xN, yP);
    PO = uint64_2(xP, Coordinates.y);
    NO = uint64_2(xN, Coordinates.y);
    PN = uint64_2(xP, yN);
    ON = uint64_2(Coordinates.x, yN);
    NN = uint64_2(xN, yN);
}
__host__ __device__ void getConsecutives3(uint64_3 Dimensions, uint64_3 Coordinates, uint64_3& PPP, uint64_3& OPP, uint64_3& NPP, uint64_3& POP, uint64_3& OOP, uint64_3& NOP, uint64_3& PNP, uint64_3& ONP, uint64_3& NNP, uint64_3& PPO, uint64_3& OPO, uint64_3& NPO, uint64_3& POO, uint64_3& NOO, uint64_3& PNO, uint64_3& ONO, uint64_3& NNO, uint64_3& PPN, uint64_3& OPN, uint64_3& NPN, uint64_3& PON, uint64_3& OON, uint64_3& NON, uint64_3& PNN, uint64_3& ONN, uint64_3& NNN) {
    uint32_t xP;
    uint32_t xN;
    uint32_t yP;
    uint32_t yN;
    uint32_t zP;
    uint32_t zN;

    getNewCoordinates3(Dimensions, Coordinates, xP, xN, yP, yN, zP, zN);

    PPP = uint64_3(xP, yP, zP);
    OPP = uint64_3(Coordinates.x, yP, zP);
    NPP = uint64_3(xN, yP, zP);
    POP = uint64_3(xP, Coordinates.y, zP);
    OOP = uint64_3(Coordinates.x, Coordinates.y, zP);
    NOP = uint64_3(xN, Coordinates.y, zP);
    PNP = uint64_3(xP, yN, zP);
    ONP = uint64_3(Coordinates.x, yN, zP);
    NNP = uint64_3(xN, yN, zP);
    PPO = uint64_3(xP, yP, Coordinates.z);
    OPO = uint64_3(Coordinates.x, yP, Coordinates.z);
    NPO = uint64_3(xN, yP, Coordinates.z);
    POO = uint64_3(xP, Coordinates.y, Coordinates.z);
    NOO = uint64_3(xN, Coordinates.y, Coordinates.z);
    PNO = uint64_3(xP, yN, Coordinates.z);
    ONO = uint64_3(Coordinates.x, yN, Coordinates.z);
    NNO = uint64_3(xN, yN, Coordinates.z);
    PPN = uint64_3(xP, yP, zN);
    OPN = uint64_3(Coordinates.x, yP, zN);
    NPN = uint64_3(xN, yP, zN);
    PON = uint64_3(xP, Coordinates.y, zN);
    OON = uint64_3(Coordinates.x, Coordinates.y, zN);
    NON = uint64_3(xN, Coordinates.y, zN);
    PNN = uint64_3(xP, yN, zN);
    ONN = uint64_3(Coordinates.x, yN, zN);
    NNN = uint64_3(xN, yN, zN);
}

__host__ __device__ void BrendanCUDA::GetConsecutives2(uint32_2 Dimensions, uint32_2 Coordinates, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) {
    getConsecutives2(Dimensions, Coordinates, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint32_3 Dimensions, uint32_3 Coordinates, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) {
    getConsecutives3(Dimensions, Coordinates, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint32_2 Dimensions, uint32_2 Coordinates, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) {
    getConsecutives2(Dimensions, Coordinates, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint32_3 Dimensions, uint32_3 Coordinates, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) {
    getConsecutives3(Dimensions, Coordinates, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index32ToCoordinates32_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index32ToCoordinates32_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index32ToCoordinates32_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index32ToCoordinates32_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint32_2 Dimensions, uint32_t Index, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) {
    getConsecutives2(Dimensions, BrendanCUDA::Index32ToCoordinates32_2_RM(Dimensions, Index), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint32_2 Dimensions, uint32_t Index, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) {
    getConsecutives2(Dimensions, BrendanCUDA::Index32ToCoordinates32_2_RM(Dimensions, Index), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex32_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex32_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex32_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex32_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index32ToCoordinates32_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index32ToCoordinates32_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index32ToCoordinates32_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index32ToCoordinates32_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint32_3 Dimensions, uint32_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) {
    getConsecutives3(Dimensions, BrendanCUDA::Index32ToCoordinates32_3_RM(Dimensions, Index), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint32_3 Dimensions, uint32_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) {
    getConsecutives3(Dimensions, BrendanCUDA::Index32ToCoordinates32_3_RM(Dimensions, Index), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex32_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex32_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex32_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex32_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index32ToCoordinates64_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index32ToCoordinates64_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint32_t Index, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index32ToCoordinates64_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint32_t Index, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index32ToCoordinates64_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint64_2 Dimensions, uint32_t Index, uint64_2& PO, uint64_2& NO, uint64_2& OP, uint64_2& ON) {
    getConsecutives2(Dimensions, BrendanCUDA::Index32ToCoordinates64_2_RM(Dimensions, Index), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint64_2 Dimensions, uint32_t Index, uint64_2& PP, uint64_2& OP, uint64_2& NP, uint64_2& PO, uint64_2& NO, uint64_2& PN, uint64_2& ON, uint64_2& NN) {
    getConsecutives2(Dimensions, BrendanCUDA::Index32ToCoordinates64_2_RM(Dimensions, Index), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex32_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex32_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t& PO, uint32_t& NO, uint32_t& OP, uint32_t& ON) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex32_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint32_t& PP, uint32_t& OP, uint32_t& NP, uint32_t& PO, uint32_t& NO, uint32_t& PN, uint32_t& ON, uint32_t& NN) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex32_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index32ToCoordinates64_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index32ToCoordinates64_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index32ToCoordinates64_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index32ToCoordinates64_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint64_3 Dimensions, uint32_t Index, uint64_3& POO, uint64_3& NOO, uint64_3& OPO, uint64_3& ONO, uint64_3& OOP, uint64_3& OON) {
    getConsecutives3(Dimensions, BrendanCUDA::Index32ToCoordinates64_3_RM(Dimensions, Index), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint64_3 Dimensions, uint32_t Index, uint64_3& PPP, uint64_3& OPP, uint64_3& NPP, uint64_3& POP, uint64_3& OOP, uint64_3& NOP, uint64_3& PNP, uint64_3& ONP, uint64_3& NNP, uint64_3& PPO, uint64_3& OPO, uint64_3& NPO, uint64_3& POO, uint64_3& NOO, uint64_3& PNO, uint64_3& ONO, uint64_3& NNO, uint64_3& PPN, uint64_3& OPN, uint64_3& NPN, uint64_3& PON, uint64_3& OON, uint64_3& NON, uint64_3& PNN, uint64_3& ONN, uint64_3& NNN) {
    getConsecutives3(Dimensions, BrendanCUDA::Index32ToCoordinates64_3_RM(Dimensions, Index), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex32_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex32_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex32_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex32_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint64_2 Dimensions, uint64_2 Coordinates, uint64_2& PO, uint64_2& NO, uint64_2& OP, uint64_2& ON) {
    getConsecutives2(Dimensions, Coordinates, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint64_3 Dimensions, uint64_3 Coordinates, uint64_3& POO, uint64_3& NOO, uint64_3& OPO, uint64_3& ONO, uint64_3& OOP, uint64_3& OON) {
    getConsecutives3(Dimensions, Coordinates, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint64_2 Dimensions, uint64_2 Coordinates, uint64_2& PP, uint64_2& OP, uint64_2& NP, uint64_2& PO, uint64_2& NO, uint64_2& PN, uint64_2& ON, uint64_2& NN) {
    getConsecutives2(Dimensions, Coordinates, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint64_3 Dimensions, uint64_3 Coordinates, uint64_3& PPP, uint64_3& OPP, uint64_3& NPP, uint64_3& POP, uint64_3& OOP, uint64_3& NOP, uint64_3& PNP, uint64_3& ONP, uint64_3& NNP, uint64_3& PPO, uint64_3& OPO, uint64_3& NPO, uint64_3& POO, uint64_3& NOO, uint64_3& PNO, uint64_3& ONO, uint64_3& NNO, uint64_3& PPN, uint64_3& OPN, uint64_3& NPN, uint64_3& PON, uint64_3& OON, uint64_3& NON, uint64_3& PNN, uint64_3& ONN, uint64_3& NNN) {
    getConsecutives3(Dimensions, Coordinates, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index64ToCoordinates32_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index64ToCoordinates32_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index64ToCoordinates32_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index64ToCoordinates32_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint32_2 Dimensions, uint64_t Index, uint32_2& PO, uint32_2& NO, uint32_2& OP, uint32_2& ON) {
    getConsecutives2(Dimensions, BrendanCUDA::Index64ToCoordinates32_2_RM(Dimensions, Index), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint32_2 Dimensions, uint64_t Index, uint32_2& PP, uint32_2& OP, uint32_2& NP, uint32_2& PO, uint32_2& NO, uint32_2& PN, uint32_2& ON, uint32_2& NN) {
    getConsecutives2(Dimensions, BrendanCUDA::Index64ToCoordinates32_2_RM(Dimensions, Index), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex64_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex64_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex64_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint32_2 Dimensions, uint32_2 Coordinates, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_2ToIndex64_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index64ToCoordinates32_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index64ToCoordinates32_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index64ToCoordinates32_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index64ToCoordinates32_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint32_3 Dimensions, uint64_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON) {
    getConsecutives3(Dimensions, BrendanCUDA::Index64ToCoordinates32_3_RM(Dimensions, Index), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint32_3 Dimensions, uint64_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN) {
    getConsecutives3(Dimensions, BrendanCUDA::Index64ToCoordinates32_3_RM(Dimensions, Index), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex64_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex64_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex64_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates32_3ToIndex64_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index64ToCoordinates64_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_RM(Dimensions, BrendanCUDA::Index64ToCoordinates64_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint64_t Index, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index64ToCoordinates64_2_RM(Dimensions, Index), Index, PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint64_t Index, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_CM(Dimensions, BrendanCUDA::Index64ToCoordinates64_2_RM(Dimensions, Index), Index, PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint64_2 Dimensions, uint64_t Index, uint64_2& PO, uint64_2& NO, uint64_2& OP, uint64_2& ON) {
    getConsecutives2(Dimensions, BrendanCUDA::Index64ToCoordinates64_2_RM(Dimensions, Index), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2(uint64_2 Dimensions, uint64_t Index, uint64_2& PP, uint64_2& OP, uint64_2& NP, uint64_2& PO, uint64_2& NO, uint64_2& PN, uint64_2& ON, uint64_2& NN) {
    getConsecutives2(Dimensions, BrendanCUDA::Index64ToCoordinates64_2_RM(Dimensions, Index), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex64_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_RM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex64_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t& PO, uint64_t& NO, uint64_t& OP, uint64_t& ON) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex64_RM(Dimensions, Coordinates), PO, NO, OP, ON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives2_CM(uint64_2 Dimensions, uint64_2 Coordinates, uint64_t& PP, uint64_t& OP, uint64_t& NP, uint64_t& PO, uint64_t& NO, uint64_t& PN, uint64_t& ON, uint64_t& NN) {
    getConsecutives2_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_2ToIndex64_RM(Dimensions, Coordinates), PP, OP, NP, PO, NO, PN, ON, NN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index64ToCoordinates64_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_RM(Dimensions, BrendanCUDA::Index64ToCoordinates64_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index64ToCoordinates64_3_RM(Dimensions, Index), Index, POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_CM(Dimensions, BrendanCUDA::Index64ToCoordinates64_3_RM(Dimensions, Index), Index, PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint64_3 Dimensions, uint64_t Index, uint64_3& POO, uint64_3& NOO, uint64_3& OPO, uint64_3& ONO, uint64_3& OOP, uint64_3& OON) {
    getConsecutives3(Dimensions, BrendanCUDA::Index64ToCoordinates64_3_RM(Dimensions, Index), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3(uint64_3 Dimensions, uint64_t Index, uint64_3& PPP, uint64_3& OPP, uint64_3& NPP, uint64_3& POP, uint64_3& OOP, uint64_3& NOP, uint64_3& PNP, uint64_3& ONP, uint64_3& NNP, uint64_3& PPO, uint64_3& OPO, uint64_3& NPO, uint64_3& POO, uint64_3& NOO, uint64_3& PNO, uint64_3& ONO, uint64_3& NNO, uint64_3& PPN, uint64_3& OPN, uint64_3& NPN, uint64_3& PON, uint64_3& OON, uint64_3& NON, uint64_3& PNN, uint64_3& ONN, uint64_3& NNN) {
    getConsecutives3(Dimensions, BrendanCUDA::Index64ToCoordinates64_3_RM(Dimensions, Index), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex64_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_RM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_RM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex64_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex64_RM(Dimensions, Coordinates), POO, NOO, OPO, ONO, OOP, OON);
}
__host__ __device__ void BrendanCUDA::GetConsecutives3_CM(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN) {
    getConsecutives3_CM(Dimensions, Coordinates, BrendanCUDA::Coordinates64_3ToIndex64_RM(Dimensions, Coordinates), PPP, OPP, NPP, POP, OOP, NOP, PNP, ONP, NNP, PPO, OPO, NPO, POO, NOO, PNO, ONO, NNO, PPN, OPN, NPN, PON, OON, NON, PNN, ONN, NNN);
}
