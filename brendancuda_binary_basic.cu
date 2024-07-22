#include "brendancuda_binary_basic.h"

__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsF(uint64_t n) {
    if (n >> 32) {
        if (n >> 48) {
            if (n >> 56) {
                if (n >> 60) {
                    if (n >> 62) {
                        if (n >> 63) {
                            return 64ui32;
                        }
                        else {
                            return 63ui32;
                        }
                    }
                    else {
                        if (n >> 61) {
                            return 62ui32;
                        }
                        else {
                            return 61ui32;
                        }
                    }
                }
                else {
                    if (n >> 58) {
                        if (n >> 59) {
                            return 60ui32;
                        }
                        else {
                            return 59ui32;
                        }
                    }
                    else {
                        if (n >> 57) {
                            return 58ui32;
                        }
                        else {
                            return 57ui32;
                        }
                    }
                }
            }
            else {
                if (n >> 52) {
                    if (n >> 54) {
                        if (n >> 55) {
                            return 56ui32;
                        }
                        else {
                            return 55ui32;
                        }
                    }
                    else {
                        if (n >> 53) {
                            return 54ui32;
                        }
                        else {
                            return 53ui32;
                        }
                    }
                }
                else {
                    if (n >> 50) {
                        if (n >> 51) {
                            return 52ui32;
                        }
                        else {
                            return 51ui32;
                        }
                    }
                    else {
                        if (n >> 49) {
                            return 50ui32;
                        }
                        else {
                            return 49ui32;
                        }
                    }
                }
            }
        }
        else {
            if (n >> 40) {
                if (n >> 44) {
                    if (n >> 46) {
                        if (n >> 47) {
                            return 48ui32;
                        }
                        else {
                            return 47ui32;
                        }
                    }
                    else {
                        if (n >> 45) {
                            return 46ui32;
                        }
                        else {
                            return 45ui32;
                        }
                    }
                }
                else {
                    if (n >> 42) {
                        if (n >> 43) {
                            return 44ui32;
                        }
                        else {
                            return 43ui32;
                        }
                    }
                    else {
                        if (n >> 41) {
                            return 42ui32;
                        }
                        else {
                            return 41ui32;
                        }
                    }
                }
            }
            else {
                if (n >> 36) {
                    if (n >> 38) {
                        if (n >> 39) {
                            return 40ui32;
                        }
                        else {
                            return 39ui32;
                        }
                    }
                    else {
                        if (n >> 37) {
                            return 38ui32;
                        }
                        else {
                            return 37ui32;
                        }
                    }
                }
                else {
                    if (n >> 34) {
                        if (n >> 35) {
                            return 36ui32;
                        }
                        else {
                            return 35ui32;
                        }
                    }
                    else {
                        if (n >> 33) {
                            return 34ui32;
                        }
                        else {
                            return 33ui32;
                        }
                    }
                }
            }
        }
    }
    else {
        if (n >> 16) {
            if (n >> 24) {
                if (n >> 28) {
                    if (n >> 30) {
                        if (n >> 31) {
                            return 32ui32;
                        }
                        else {
                            return 31ui32;
                        }
                    }
                    else {
                        if (n >> 29) {
                            return 30ui32;
                        }
                        else {
                            return 29ui32;
                        }
                    }
                }
                else {
                    if (n >> 26) {
                        if (n >> 27) {
                            return 28ui32;
                        }
                        else {
                            return 27ui32;
                        }
                    }
                    else {
                        if (n >> 25) {
                            return 26ui32;
                        }
                        else {
                            return 25ui32;
                        }
                    }
                }
            }
            else {
                if (n >> 20) {
                    if (n >> 22) {
                        if (n >> 23) {
                            return 24ui32;
                        }
                        else {
                            return 23ui32;
                        }
                    }
                    else {
                        if (n >> 21) {
                            return 22ui32;
                        }
                        else {
                            return 21ui32;
                        }
                    }
                }
                else {
                    if (n >> 18) {
                        if (n >> 19) {
                            return 20ui32;
                        }
                        else {
                            return 19ui32;
                        }
                    }
                    else {
                        if (n >> 17) {
                            return 18ui32;
                        }
                        else {
                            return 17ui32;
                        }
                    }
                }
            }
        }
        else {
            if (n >> 8) {
                if (n >> 12) {
                    if (n >> 14) {
                        if (n >> 15) {
                            return 16ui32;
                        }
                        else {
                            return 15ui32;
                        }
                    }
                    else {
                        if (n >> 13) {
                            return 14ui32;
                        }
                        else {
                            return 13ui32;
                        }
                    }
                }
                else {
                    if (n >> 10) {
                        if (n >> 11) {
                            return 12ui32;
                        }
                        else {
                            return 11ui32;
                        }
                    }
                    else {
                        if (n >> 9) {
                            return 10ui32;
                        }
                        else {
                            return 9ui32;
                        }
                    }
                }
            }
            else {
                if (n >> 4) {
                    if (n >> 6) {
                        if (n >> 7) {
                            return 8ui32;
                        }
                        else {
                            return 7ui32;
                        }
                    }
                    else {
                        if (n >> 5) {
                            return 6ui32;
                        }
                        else {
                            return 5ui32;
                        }
                    }
                }
                else {
                    if (n >> 2) {
                        if (n >> 3) {
                            return 4ui32;
                        }
                        else {
                            return 3ui32;
                        }
                    }
                    else {
                        if (n >> 1) {
                            return 2ui32;
                        }
                        else {
                            if (n) {
                                return 1ui32;
                            }
                            else {
                                return 0ui32;
                            }
                        }
                    }
                }
            }
        }
    }
}
__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsF(uint32_t n) {
    if (n >> 16) {
        if (n >> 24) {
            if (n >> 28) {
                if (n >> 30) {
                    if (n >> 31) {
                        return 32ui32;
                    }
                    else {
                        return 31ui32;
                    }
                }
                else {
                    if (n >> 29) {
                        return 30ui32;
                    }
                    else {
                        return 29ui32;
                    }
                }
            }
            else {
                if (n >> 26) {
                    if (n >> 27) {
                        return 28ui32;
                    }
                    else {
                        return 27ui32;
                    }
                }
                else {
                    if (n >> 25) {
                        return 26ui32;
                    }
                    else {
                        return 25ui32;
                    }
                }
            }
        }
        else {
            if (n >> 20) {
                if (n >> 22) {
                    if (n >> 23) {
                        return 24ui32;
                    }
                    else {
                        return 23ui32;
                    }
                }
                else {
                    if (n >> 21) {
                        return 22ui32;
                    }
                    else {
                        return 21ui32;
                    }
                }
            }
            else {
                if (n >> 18) {
                    if (n >> 19) {
                        return 20ui32;
                    }
                    else {
                        return 19ui32;
                    }
                }
                else {
                    if (n >> 17) {
                        return 18ui32;
                    }
                    else {
                        return 17ui32;
                    }
                }
            }
        }
    }
    else {
        if (n >> 8) {
            if (n >> 12) {
                if (n >> 14) {
                    if (n >> 15) {
                        return 16ui32;
                    }
                    else {
                        return 15ui32;
                    }
                }
                else {
                    if (n >> 13) {
                        return 14ui32;
                    }
                    else {
                        return 13ui32;
                    }
                }
            }
            else {
                if (n >> 10) {
                    if (n >> 11) {
                        return 12ui32;
                    }
                    else {
                        return 11ui32;
                    }
                }
                else {
                    if (n >> 9) {
                        return 10ui32;
                    }
                    else {
                        return 9ui32;
                    }
                }
            }
        }
        else {
            if (n >> 4) {
                if (n >> 6) {
                    if (n >> 7) {
                        return 8ui32;
                    }
                    else {
                        return 7ui32;
                    }
                }
                else {
                    if (n >> 5) {
                        return 6ui32;
                    }
                    else {
                        return 5ui32;
                    }
                }
            }
            else {
                if (n >> 2) {
                    if (n >> 3) {
                        return 4ui32;
                    }
                    else {
                        return 3ui32;
                    }
                }
                else {
                    if (n >> 1) {
                        return 2ui32;
                    }
                    else {
                        if (n) {
                            return 1ui32;
                        }
                        else {
                            return 0ui32;
                        }
                    }
                }
            }
        }
    }
}
__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsF(uint16_t n) {
    uint32_t ni = (uint32_t)n;
    if (ni >> 8) {
        if (ni >> 12) {
            if (ni >> 14) {
                if (ni >> 15) {
                    return 16ui32;
                }
                else {
                    return 15ui32;
                }
            }
            else {
                if (ni >> 13) {
                    return 14ui32;
                }
                else {
                    return 13ui32;
                }
            }
        }
        else {
            if (ni >> 10) {
                if (ni >> 11) {
                    return 12ui32;
                }
                else {
                    return 11ui32;
                }
            }
            else {
                if (ni >> 9) {
                    return 10ui32;
                }
                else {
                    return 9ui32;
                }
            }
        }
    }
    else {
        if (ni >> 4) {
            if (ni >> 6) {
                if (ni >> 7) {
                    return 8ui32;
                }
                else {
                    return 7ui32;
                }
            }
            else {
                if (ni >> 5) {
                    return 6ui32;
                }
                else {
                    return 5ui32;
                }
            }
        }
        else {
            if (ni >> 2) {
                if (ni >> 3) {
                    return 4ui32;
                }
                else {
                    return 3ui32;
                }
            }
            else {
                if (ni >> 1) {
                    return 2ui32;
                }
                else {
                    if (ni) {
                        return 1ui32;
                    }
                    else {
                        return 0ui32;
                    }
                }
            }
        }
    }
}
__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsF(uint8_t n) {
    uint32_t ni = (uint32_t)n;
    if (ni >> 4) {
        if (ni >> 6) {
            if (ni >> 7) {
                return 8ui32;
            }
            else {
                return 7ui32;
            }
        }
        else {
            if (ni >> 5) {
                return 6ui32;
            }
            else {
                return 5ui32;
            }
        }
    }
    else {
        if (ni >> 2) {
            if (ni >> 3) {
                return 4ui32;
            }
            else {
                return 3ui32;
            }
        }
        else {
            if (ni >> 1) {
                return 2ui32;
            }
            else {
                if (ni) {
                    return 1ui32;
                }
                else {
                    return 0ui32;
                }
            }
        }
    }
}
__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsB(uint64_t n) {
    if (n << 32) {
        if (n << 48) {
            if (n << 56) {
                if (n << 60) {
                    if (n << 62) {
                        if (n << 63) {
                            return 64ui32;
                        }
                        else {
                            return 63ui32;
                        }
                    }
                    else {
                        if (n << 61) {
                            return 62ui32;
                        }
                        else {
                            return 61ui32;
                        }
                    }
                }
                else {
                    if (n << 58) {
                        if (n << 59) {
                            return 60ui32;
                        }
                        else {
                            return 59ui32;
                        }
                    }
                    else {
                        if (n << 57) {
                            return 58ui32;
                        }
                        else {
                            return 57ui32;
                        }
                    }
                }
            }
            else {
                if (n << 52) {
                    if (n << 54) {
                        if (n << 55) {
                            return 56ui32;
                        }
                        else {
                            return 55ui32;
                        }
                    }
                    else {
                        if (n << 53) {
                            return 54ui32;
                        }
                        else {
                            return 53ui32;
                        }
                    }
                }
                else {
                    if (n << 50) {
                        if (n << 51) {
                            return 52ui32;
                        }
                        else {
                            return 51ui32;
                        }
                    }
                    else {
                        if (n << 49) {
                            return 50ui32;
                        }
                        else {
                            return 49ui32;
                        }
                    }
                }
            }
        }
        else {
            if (n << 40) {
                if (n << 44) {
                    if (n << 46) {
                        if (n << 47) {
                            return 48ui32;
                        }
                        else {
                            return 47ui32;
                        }
                    }
                    else {
                        if (n << 45) {
                            return 46ui32;
                        }
                        else {
                            return 45ui32;
                        }
                    }
                }
                else {
                    if (n << 42) {
                        if (n << 43) {
                            return 44ui32;
                        }
                        else {
                            return 43ui32;
                        }
                    }
                    else {
                        if (n << 41) {
                            return 42ui32;
                        }
                        else {
                            return 41ui32;
                        }
                    }
                }
            }
            else {
                if (n << 36) {
                    if (n << 38) {
                        if (n << 39) {
                            return 40ui32;
                        }
                        else {
                            return 39ui32;
                        }
                    }
                    else {
                        if (n << 37) {
                            return 38ui32;
                        }
                        else {
                            return 37ui32;
                        }
                    }
                }
                else {
                    if (n << 34) {
                        if (n << 35) {
                            return 36ui32;
                        }
                        else {
                            return 35ui32;
                        }
                    }
                    else {
                        if (n << 33) {
                            return 34ui32;
                        }
                        else {
                            return 33ui32;
                        }
                    }
                }
            }
        }
    }
    else {
        if (n << 16) {
            if (n << 24) {
                if (n << 28) {
                    if (n << 30) {
                        if (n << 31) {
                            return 32ui32;
                        }
                        else {
                            return 31ui32;
                        }
                    }
                    else {
                        if (n << 29) {
                            return 30ui32;
                        }
                        else {
                            return 29ui32;
                        }
                    }
                }
                else {
                    if (n << 26) {
                        if (n << 27) {
                            return 28ui32;
                        }
                        else {
                            return 27ui32;
                        }
                    }
                    else {
                        if (n << 25) {
                            return 26ui32;
                        }
                        else {
                            return 25ui32;
                        }
                    }
                }
            }
            else {
                if (n << 20) {
                    if (n << 22) {
                        if (n << 23) {
                            return 24ui32;
                        }
                        else {
                            return 23ui32;
                        }
                    }
                    else {
                        if (n << 21) {
                            return 22ui32;
                        }
                        else {
                            return 21ui32;
                        }
                    }
                }
                else {
                    if (n << 18) {
                        if (n << 19) {
                            return 20ui32;
                        }
                        else {
                            return 19ui32;
                        }
                    }
                    else {
                        if (n << 17) {
                            return 18ui32;
                        }
                        else {
                            return 17ui32;
                        }
                    }
                }
            }
        }
        else {
            if (n << 8) {
                if (n << 12) {
                    if (n << 14) {
                        if (n << 15) {
                            return 16ui32;
                        }
                        else {
                            return 15ui32;
                        }
                    }
                    else {
                        if (n << 13) {
                            return 14ui32;
                        }
                        else {
                            return 13ui32;
                        }
                    }
                }
                else {
                    if (n << 10) {
                        if (n << 11) {
                            return 12ui32;
                        }
                        else {
                            return 11ui32;
                        }
                    }
                    else {
                        if (n << 9) {
                            return 10ui32;
                        }
                        else {
                            return 9ui32;
                        }
                    }
                }
            }
            else {
                if (n << 4) {
                    if (n << 6) {
                        if (n << 7) {
                            return 8ui32;
                        }
                        else {
                            return 7ui32;
                        }
                    }
                    else {
                        if (n << 5) {
                            return 6ui32;
                        }
                        else {
                            return 5ui32;
                        }
                    }
                }
                else {
                    if (n << 2) {
                        if (n << 3) {
                            return 4ui32;
                        }
                        else {
                            return 3ui32;
                        }
                    }
                    else {
                        if (n << 1) {
                            return 2ui32;
                        }
                        else {
                            if (n) {
                                return 1ui32;
                            }
                            else {
                                return 0ui32;
                            }
                        }
                    }
                }
            }
        }
    }
}
__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsB(uint32_t n) {
    if (n << 16) {
        if (n << 24) {
            if (n << 28) {
                if (n << 30) {
                    if (n << 31) {
                        return 32ui32;
                    }
                    else {
                        return 31ui32;
                    }
                }
                else {
                    if (n << 29) {
                        return 30ui32;
                    }
                    else {
                        return 29ui32;
                    }
                }
            }
            else {
                if (n << 26) {
                    if (n << 27) {
                        return 28ui32;
                    }
                    else {
                        return 27ui32;
                    }
                }
                else {
                    if (n << 25) {
                        return 26ui32;
                    }
                    else {
                        return 25ui32;
                    }
                }
            }
        }
        else {
            if (n << 20) {
                if (n << 22) {
                    if (n << 23) {
                        return 24ui32;
                    }
                    else {
                        return 23ui32;
                    }
                }
                else {
                    if (n << 21) {
                        return 22ui32;
                    }
                    else {
                        return 21ui32;
                    }
                }
            }
            else {
                if (n << 18) {
                    if (n << 19) {
                        return 20ui32;
                    }
                    else {
                        return 19ui32;
                    }
                }
                else {
                    if (n << 17) {
                        return 18ui32;
                    }
                    else {
                        return 17ui32;
                    }
                }
            }
        }
    }
    else {
        if (n << 8) {
            if (n << 12) {
                if (n << 14) {
                    if (n << 15) {
                        return 16ui32;
                    }
                    else {
                        return 15ui32;
                    }
                }
                else {
                    if (n << 13) {
                        return 14ui32;
                    }
                    else {
                        return 13ui32;
                    }
                }
            }
            else {
                if (n << 10) {
                    if (n << 11) {
                        return 12ui32;
                    }
                    else {
                        return 11ui32;
                    }
                }
                else {
                    if (n << 9) {
                        return 10ui32;
                    }
                    else {
                        return 9ui32;
                    }
                }
            }
        }
        else {
            if (n << 4) {
                if (n << 6) {
                    if (n << 7) {
                        return 8ui32;
                    }
                    else {
                        return 7ui32;
                    }
                }
                else {
                    if (n << 5) {
                        return 6ui32;
                    }
                    else {
                        return 5ui32;
                    }
                }
            }
            else {
                if (n << 2) {
                    if (n << 3) {
                        return 4ui32;
                    }
                    else {
                        return 3ui32;
                    }
                }
                else {
                    if (n << 1) {
                        return 2ui32;
                    }
                    else {
                        if (n) {
                            return 1ui32;
                        }
                        else {
                            return 0ui32;
                        }
                    }
                }
            }
        }
    }
}
__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsB(uint16_t n) {
    uint32_t ni = (uint32_t)n;
    if (ni << 8) {
        if (ni << 12) {
            if (ni << 14) {
                if (ni << 15) {
                    return 16ui32;
                }
                else {
                    return 15ui32;
                }
            }
            else {
                if (ni << 13) {
                    return 14ui32;
                }
                else {
                    return 13ui32;
                }
            }
        }
        else {
            if (ni << 10) {
                if (ni << 11) {
                    return 12ui32;
                }
                else {
                    return 11ui32;
                }
            }
            else {
                if (ni << 9) {
                    return 10ui32;
                }
                else {
                    return 9ui32;
                }
            }
        }
    }
    else {
        if (ni << 4) {
            if (ni << 6) {
                if (ni << 7) {
                    return 8ui32;
                }
                else {
                    return 7ui32;
                }
            }
            else {
                if (ni << 5) {
                    return 6ui32;
                }
                else {
                    return 5ui32;
                }
            }
        }
        else {
            if (ni << 2) {
                if (ni << 3) {
                    return 4ui32;
                }
                else {
                    return 3ui32;
                }
            }
            else {
                if (ni << 1) {
                    return 2ui32;
                }
                else {
                    if (ni) {
                        return 1ui32;
                    }
                    else {
                        return 0ui32;
                    }
                }
            }
        }
    }
}
__host__ __device__ uint32_t BrendanCUDA::Binary::CountBitsB(uint8_t n) {
    uint32_t ni = (uint32_t)n;
    if (ni << 4) {
        if (ni << 6) {
            if (ni << 7) {
                return 8ui32;
            }
            else {
                return 7ui32;
            }
        }
        else {
            if (ni << 5) {
                return 6ui32;
            }
            else {
                return 5ui32;
            }
        }
    }
    else {
        if (ni << 2) {
            if (ni << 3) {
                return 4ui32;
            }
            else {
                return 3ui32;
            }
        }
        else {
            if (ni << 1) {
                return 2ui32;
            }
            else {
                if (ni) {
                    return 1ui32;
                }
                else {
                    return 0ui32;
                }
            }
        }
    }
}