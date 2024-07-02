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
__host__ __device__ uint32_t BrendanCUDA::Binary::Count1s(uint64_t Value) {
    uint32_t c = 0;
    if (Value & (1ui64 << 0)) {
        ++c;
    }
    if (Value & (1ui64 << 1)) {
        ++c;
    }
    if (Value & (1ui64 << 2)) {
        ++c;
    }
    if (Value & (1ui64 << 3)) {
        ++c;
    }
    if (Value & (1ui64 << 4)) {
        ++c;
    }
    if (Value & (1ui64 << 5)) {
        ++c;
    }
    if (Value & (1ui64 << 6)) {
        ++c;
    }
    if (Value & (1ui64 << 7)) {
        ++c;
    }
    if (Value & (1ui64 << 8)) {
        ++c;
    }
    if (Value & (1ui64 << 9)) {
        ++c;
    }
    if (Value & (1ui64 << 10)) {
        ++c;
    }
    if (Value & (1ui64 << 11)) {
        ++c;
    }
    if (Value & (1ui64 << 12)) {
        ++c;
    }
    if (Value & (1ui64 << 13)) {
        ++c;
    }
    if (Value & (1ui64 << 14)) {
        ++c;
    }
    if (Value & (1ui64 << 15)) {
        ++c;
    }
    if (Value & (1ui64 << 16)) {
        ++c;
    }
    if (Value & (1ui64 << 17)) {
        ++c;
    }
    if (Value & (1ui64 << 18)) {
        ++c;
    }
    if (Value & (1ui64 << 19)) {
        ++c;
    }
    if (Value & (1ui64 << 20)) {
        ++c;
    }
    if (Value & (1ui64 << 21)) {
        ++c;
    }
    if (Value & (1ui64 << 22)) {
        ++c;
    }
    if (Value & (1ui64 << 23)) {
        ++c;
    }
    if (Value & (1ui64 << 24)) {
        ++c;
    }
    if (Value & (1ui64 << 25)) {
        ++c;
    }
    if (Value & (1ui64 << 26)) {
        ++c;
    }
    if (Value & (1ui64 << 27)) {
        ++c;
    }
    if (Value & (1ui64 << 28)) {
        ++c;
    }
    if (Value & (1ui64 << 29)) {
        ++c;
    }
    if (Value & (1ui64 << 30)) {
        ++c;
    }
    if (Value & (1ui64 << 31)) {
        ++c;
    }
    if (Value & (1ui64 << 32)) {
        ++c;
    }
    if (Value & (1ui64 << 33)) {
        ++c;
    }
    if (Value & (1ui64 << 34)) {
        ++c;
    }
    if (Value & (1ui64 << 35)) {
        ++c;
    }
    if (Value & (1ui64 << 36)) {
        ++c;
    }
    if (Value & (1ui64 << 37)) {
        ++c;
    }
    if (Value & (1ui64 << 38)) {
        ++c;
    }
    if (Value & (1ui64 << 39)) {
        ++c;
    }
    if (Value & (1ui64 << 40)) {
        ++c;
    }
    if (Value & (1ui64 << 41)) {
        ++c;
    }
    if (Value & (1ui64 << 42)) {
        ++c;
    }
    if (Value & (1ui64 << 43)) {
        ++c;
    }
    if (Value & (1ui64 << 44)) {
        ++c;
    }
    if (Value & (1ui64 << 45)) {
        ++c;
    }
    if (Value & (1ui64 << 46)) {
        ++c;
    }
    if (Value & (1ui64 << 47)) {
        ++c;
    }
    if (Value & (1ui64 << 48)) {
        ++c;
    }
    if (Value & (1ui64 << 49)) {
        ++c;
    }
    if (Value & (1ui64 << 50)) {
        ++c;
    }
    if (Value & (1ui64 << 51)) {
        ++c;
    }
    if (Value & (1ui64 << 52)) {
        ++c;
    }
    if (Value & (1ui64 << 53)) {
        ++c;
    }
    if (Value & (1ui64 << 54)) {
        ++c;
    }
    if (Value & (1ui64 << 55)) {
        ++c;
    }
    if (Value & (1ui64 << 56)) {
        ++c;
    }
    if (Value & (1ui64 << 57)) {
        ++c;
    }
    if (Value & (1ui64 << 58)) {
        ++c;
    }
    if (Value & (1ui64 << 59)) {
        ++c;
    }
    if (Value & (1ui64 << 60)) {
        ++c;
    }
    if (Value & (1ui64 << 61)) {
        ++c;
    }
    if (Value & (1ui64 << 62)) {
        ++c;
    }
    if (Value & (1ui64 << 63)) {
        ++c;
    }
    return c;
}
__host__ __device__ uint32_t BrendanCUDA::Binary::Count1s(uint32_t Value) {
    uint32_t c = 0;
    if (Value & (1ui32 << 0)) {
        ++c;
    }
    if (Value & (1ui32 << 1)) {
        ++c;
    }
    if (Value & (1ui32 << 2)) {
        ++c;
    }
    if (Value & (1ui32 << 3)) {
        ++c;
    }
    if (Value & (1ui32 << 4)) {
        ++c;
    }
    if (Value & (1ui32 << 5)) {
        ++c;
    }
    if (Value & (1ui32 << 6)) {
        ++c;
    }
    if (Value & (1ui32 << 7)) {
        ++c;
    }
    if (Value & (1ui32 << 8)) {
        ++c;
    }
    if (Value & (1ui32 << 9)) {
        ++c;
    }
    if (Value & (1ui32 << 10)) {
        ++c;
    }
    if (Value & (1ui32 << 11)) {
        ++c;
    }
    if (Value & (1ui32 << 12)) {
        ++c;
    }
    if (Value & (1ui32 << 13)) {
        ++c;
    }
    if (Value & (1ui32 << 14)) {
        ++c;
    }
    if (Value & (1ui32 << 15)) {
        ++c;
    }
    if (Value & (1ui32 << 16)) {
        ++c;
    }
    if (Value & (1ui32 << 17)) {
        ++c;
    }
    if (Value & (1ui32 << 18)) {
        ++c;
    }
    if (Value & (1ui32 << 19)) {
        ++c;
    }
    if (Value & (1ui32 << 20)) {
        ++c;
    }
    if (Value & (1ui32 << 21)) {
        ++c;
    }
    if (Value & (1ui32 << 22)) {
        ++c;
    }
    if (Value & (1ui32 << 23)) {
        ++c;
    }
    if (Value & (1ui32 << 24)) {
        ++c;
    }
    if (Value & (1ui32 << 25)) {
        ++c;
    }
    if (Value & (1ui32 << 26)) {
        ++c;
    }
    if (Value & (1ui32 << 27)) {
        ++c;
    }
    if (Value & (1ui32 << 28)) {
        ++c;
    }
    if (Value & (1ui32 << 29)) {
        ++c;
    }
    if (Value & (1ui32 << 30)) {
        ++c;
    }
    if (Value & (1ui32 << 31)) {
        ++c;
    }
    return c;
}
__host__ __device__ uint32_t BrendanCUDA::Binary::Count1s(uint16_t Value) {
    uint32_t c = 0;
    if (Value & (1ui16 << 0)) {
        ++c;
    }
    if (Value & (1ui16 << 1)) {
        ++c;
    }
    if (Value & (1ui16 << 2)) {
        ++c;
    }
    if (Value & (1ui16 << 3)) {
        ++c;
    }
    if (Value & (1ui16 << 4)) {
        ++c;
    }
    if (Value & (1ui16 << 5)) {
        ++c;
    }
    if (Value & (1ui16 << 6)) {
        ++c;
    }
    if (Value & (1ui16 << 7)) {
        ++c;
    }
    if (Value & (1ui16 << 8)) {
        ++c;
    }
    if (Value & (1ui16 << 9)) {
        ++c;
    }
    if (Value & (1ui16 << 10)) {
        ++c;
    }
    if (Value & (1ui16 << 11)) {
        ++c;
    }
    if (Value & (1ui16 << 12)) {
        ++c;
    }
    if (Value & (1ui16 << 13)) {
        ++c;
    }
    if (Value & (1ui16 << 14)) {
        ++c;
    }
    if (Value & (1ui16 << 15)) {
        ++c;
    }
    return c;
}
__host__ __device__ uint32_t BrendanCUDA::Binary::Count1s(uint8_t Value) {
    uint32_t c = 0;
    if (Value & (1ui8 << 0)) {
        ++c;
    }
    if (Value & (1ui8 << 1)) {
        ++c;
    }
    if (Value & (1ui8 << 2)) {
        ++c;
    }
    if (Value & (1ui8 << 3)) {
        ++c;
    }
    if (Value & (1ui8 << 4)) {
        ++c;
    }
    if (Value & (1ui8 << 5)) {
        ++c;
    }
    if (Value & (1ui8 << 6)) {
        ++c;
    }
    if (Value & (1ui8 << 7)) {
        ++c;
    }
    return c;
}