#include "brendancuda_ai.h"
#include "brendancuda_errorhelp.h"
#include <curand_kernel.h>

__host__ __device__ void copyFloatsToInt32Func(float* Floats, uint32_t* Int32, float Split) {
    uint32_t m = 0;
    if (Floats[0] > Split) m |= 1ui32 << 0;
    if (Floats[1] > Split) m |= 1ui32 << 1;
    if (Floats[2] > Split) m |= 1ui32 << 2;
    if (Floats[3] > Split) m |= 1ui32 << 3;
    if (Floats[4] > Split) m |= 1ui32 << 4;
    if (Floats[5] > Split) m |= 1ui32 << 5;
    if (Floats[6] > Split) m |= 1ui32 << 6;
    if (Floats[7] > Split) m |= 1ui32 << 7;
    if (Floats[8] > Split) m |= 1ui32 << 8;
    if (Floats[9] > Split) m |= 1ui32 << 9;
    if (Floats[10] > Split) m |= 1ui32 << 10;
    if (Floats[11] > Split) m |= 1ui32 << 11;
    if (Floats[12] > Split) m |= 1ui32 << 12;
    if (Floats[13] > Split) m |= 1ui32 << 13;
    if (Floats[14] > Split) m |= 1ui32 << 14;
    if (Floats[15] > Split) m |= 1ui32 << 15;
    if (Floats[16] > Split) m |= 1ui32 << 16;
    if (Floats[17] > Split) m |= 1ui32 << 17;
    if (Floats[18] > Split) m |= 1ui32 << 18;
    if (Floats[19] > Split) m |= 1ui32 << 19;
    if (Floats[20] > Split) m |= 1ui32 << 20;
    if (Floats[21] > Split) m |= 1ui32 << 21;
    if (Floats[22] > Split) m |= 1ui32 << 22;
    if (Floats[23] > Split) m |= 1ui32 << 23;
    if (Floats[24] > Split) m |= 1ui32 << 24;
    if (Floats[25] > Split) m |= 1ui32 << 25;
    if (Floats[26] > Split) m |= 1ui32 << 26;
    if (Floats[27] > Split) m |= 1ui32 << 27;
    if (Floats[28] > Split) m |= 1ui32 << 28;
    if (Floats[29] > Split) m |= 1ui32 << 29;
    if (Floats[30] > Split) m |= 1ui32 << 30;
    if (Floats[31] > Split) m |= 1ui32 << 31;
    *Int32 = m;
}
__host__ __device__ void copyDoublesToInt32Func(double* Doubles, uint32_t* Int32, double Split) {
    uint32_t m = 0;
    if (Doubles[0] > Split) m |= 1ui32 << 0;
    if (Doubles[1] > Split) m |= 1ui32 << 1;
    if (Doubles[2] > Split) m |= 1ui32 << 2;
    if (Doubles[3] > Split) m |= 1ui32 << 3;
    if (Doubles[4] > Split) m |= 1ui32 << 4;
    if (Doubles[5] > Split) m |= 1ui32 << 5;
    if (Doubles[6] > Split) m |= 1ui32 << 6;
    if (Doubles[7] > Split) m |= 1ui32 << 7;
    if (Doubles[8] > Split) m |= 1ui32 << 8;
    if (Doubles[9] > Split) m |= 1ui32 << 9;
    if (Doubles[10] > Split) m |= 1ui32 << 10;
    if (Doubles[11] > Split) m |= 1ui32 << 11;
    if (Doubles[12] > Split) m |= 1ui32 << 12;
    if (Doubles[13] > Split) m |= 1ui32 << 13;
    if (Doubles[14] > Split) m |= 1ui32 << 14;
    if (Doubles[15] > Split) m |= 1ui32 << 15;
    if (Doubles[16] > Split) m |= 1ui32 << 16;
    if (Doubles[17] > Split) m |= 1ui32 << 17;
    if (Doubles[18] > Split) m |= 1ui32 << 18;
    if (Doubles[19] > Split) m |= 1ui32 << 19;
    if (Doubles[20] > Split) m |= 1ui32 << 20;
    if (Doubles[21] > Split) m |= 1ui32 << 21;
    if (Doubles[22] > Split) m |= 1ui32 << 22;
    if (Doubles[23] > Split) m |= 1ui32 << 23;
    if (Doubles[24] > Split) m |= 1ui32 << 24;
    if (Doubles[25] > Split) m |= 1ui32 << 25;
    if (Doubles[26] > Split) m |= 1ui32 << 26;
    if (Doubles[27] > Split) m |= 1ui32 << 27;
    if (Doubles[28] > Split) m |= 1ui32 << 28;
    if (Doubles[29] > Split) m |= 1ui32 << 29;
    if (Doubles[30] > Split) m |= 1ui32 << 30;
    if (Doubles[31] > Split) m |= 1ui32 << 31;
    *Int32 = m;
}
__host__ __device__ void copyFloatsToInt64Func(float* Floats, uint64_t* Int64, float Split) {
    uint64_t m = 0;
    if (Floats[0] > Split) m |= 1ui64 << 0;
    if (Floats[1] > Split) m |= 1ui64 << 1;
    if (Floats[2] > Split) m |= 1ui64 << 2;
    if (Floats[3] > Split) m |= 1ui64 << 3;
    if (Floats[4] > Split) m |= 1ui64 << 4;
    if (Floats[5] > Split) m |= 1ui64 << 5;
    if (Floats[6] > Split) m |= 1ui64 << 6;
    if (Floats[7] > Split) m |= 1ui64 << 7;
    if (Floats[8] > Split) m |= 1ui64 << 8;
    if (Floats[9] > Split) m |= 1ui64 << 9;
    if (Floats[10] > Split) m |= 1ui64 << 10;
    if (Floats[11] > Split) m |= 1ui64 << 11;
    if (Floats[12] > Split) m |= 1ui64 << 12;
    if (Floats[13] > Split) m |= 1ui64 << 13;
    if (Floats[14] > Split) m |= 1ui64 << 14;
    if (Floats[15] > Split) m |= 1ui64 << 15;
    if (Floats[16] > Split) m |= 1ui64 << 16;
    if (Floats[17] > Split) m |= 1ui64 << 17;
    if (Floats[18] > Split) m |= 1ui64 << 18;
    if (Floats[19] > Split) m |= 1ui64 << 19;
    if (Floats[20] > Split) m |= 1ui64 << 20;
    if (Floats[21] > Split) m |= 1ui64 << 21;
    if (Floats[22] > Split) m |= 1ui64 << 22;
    if (Floats[23] > Split) m |= 1ui64 << 23;
    if (Floats[24] > Split) m |= 1ui64 << 24;
    if (Floats[25] > Split) m |= 1ui64 << 25;
    if (Floats[26] > Split) m |= 1ui64 << 26;
    if (Floats[27] > Split) m |= 1ui64 << 27;
    if (Floats[28] > Split) m |= 1ui64 << 28;
    if (Floats[29] > Split) m |= 1ui64 << 29;
    if (Floats[30] > Split) m |= 1ui64 << 30;
    if (Floats[31] > Split) m |= 1ui64 << 31;
    if (Floats[32] > Split) m |= 1ui64 << 32;
    if (Floats[33] > Split) m |= 1ui64 << 33;
    if (Floats[34] > Split) m |= 1ui64 << 34;
    if (Floats[35] > Split) m |= 1ui64 << 35;
    if (Floats[36] > Split) m |= 1ui64 << 36;
    if (Floats[37] > Split) m |= 1ui64 << 37;
    if (Floats[38] > Split) m |= 1ui64 << 38;
    if (Floats[39] > Split) m |= 1ui64 << 39;
    if (Floats[40] > Split) m |= 1ui64 << 40;
    if (Floats[41] > Split) m |= 1ui64 << 41;
    if (Floats[42] > Split) m |= 1ui64 << 42;
    if (Floats[43] > Split) m |= 1ui64 << 43;
    if (Floats[44] > Split) m |= 1ui64 << 44;
    if (Floats[45] > Split) m |= 1ui64 << 45;
    if (Floats[46] > Split) m |= 1ui64 << 46;
    if (Floats[47] > Split) m |= 1ui64 << 47;
    if (Floats[48] > Split) m |= 1ui64 << 48;
    if (Floats[49] > Split) m |= 1ui64 << 49;
    if (Floats[50] > Split) m |= 1ui64 << 50;
    if (Floats[51] > Split) m |= 1ui64 << 51;
    if (Floats[52] > Split) m |= 1ui64 << 52;
    if (Floats[53] > Split) m |= 1ui64 << 53;
    if (Floats[54] > Split) m |= 1ui64 << 54;
    if (Floats[55] > Split) m |= 1ui64 << 55;
    if (Floats[56] > Split) m |= 1ui64 << 56;
    if (Floats[57] > Split) m |= 1ui64 << 57;
    if (Floats[58] > Split) m |= 1ui64 << 58;
    if (Floats[59] > Split) m |= 1ui64 << 59;
    if (Floats[60] > Split) m |= 1ui64 << 60;
    if (Floats[61] > Split) m |= 1ui64 << 61;
    if (Floats[62] > Split) m |= 1ui64 << 62;
    if (Floats[63] > Split) m |= 1ui64 << 63;
    *Int64 = m;
}
__host__ __device__ void copyDoublesToInt64Func(double* Doubles, uint64_t* Int64, double Split) {
    uint64_t m = 0;
    if (Doubles[0] > Split) m |= 1ui64 << 0;
    if (Doubles[1] > Split) m |= 1ui64 << 1;
    if (Doubles[2] > Split) m |= 1ui64 << 2;
    if (Doubles[3] > Split) m |= 1ui64 << 3;
    if (Doubles[4] > Split) m |= 1ui64 << 4;
    if (Doubles[5] > Split) m |= 1ui64 << 5;
    if (Doubles[6] > Split) m |= 1ui64 << 6;
    if (Doubles[7] > Split) m |= 1ui64 << 7;
    if (Doubles[8] > Split) m |= 1ui64 << 8;
    if (Doubles[9] > Split) m |= 1ui64 << 9;
    if (Doubles[10] > Split) m |= 1ui64 << 10;
    if (Doubles[11] > Split) m |= 1ui64 << 11;
    if (Doubles[12] > Split) m |= 1ui64 << 12;
    if (Doubles[13] > Split) m |= 1ui64 << 13;
    if (Doubles[14] > Split) m |= 1ui64 << 14;
    if (Doubles[15] > Split) m |= 1ui64 << 15;
    if (Doubles[16] > Split) m |= 1ui64 << 16;
    if (Doubles[17] > Split) m |= 1ui64 << 17;
    if (Doubles[18] > Split) m |= 1ui64 << 18;
    if (Doubles[19] > Split) m |= 1ui64 << 19;
    if (Doubles[20] > Split) m |= 1ui64 << 20;
    if (Doubles[21] > Split) m |= 1ui64 << 21;
    if (Doubles[22] > Split) m |= 1ui64 << 22;
    if (Doubles[23] > Split) m |= 1ui64 << 23;
    if (Doubles[24] > Split) m |= 1ui64 << 24;
    if (Doubles[25] > Split) m |= 1ui64 << 25;
    if (Doubles[26] > Split) m |= 1ui64 << 26;
    if (Doubles[27] > Split) m |= 1ui64 << 27;
    if (Doubles[28] > Split) m |= 1ui64 << 28;
    if (Doubles[29] > Split) m |= 1ui64 << 29;
    if (Doubles[30] > Split) m |= 1ui64 << 30;
    if (Doubles[31] > Split) m |= 1ui64 << 31;
    if (Doubles[32] > Split) m |= 1ui64 << 32;
    if (Doubles[33] > Split) m |= 1ui64 << 33;
    if (Doubles[34] > Split) m |= 1ui64 << 34;
    if (Doubles[35] > Split) m |= 1ui64 << 35;
    if (Doubles[36] > Split) m |= 1ui64 << 36;
    if (Doubles[37] > Split) m |= 1ui64 << 37;
    if (Doubles[38] > Split) m |= 1ui64 << 38;
    if (Doubles[39] > Split) m |= 1ui64 << 39;
    if (Doubles[40] > Split) m |= 1ui64 << 40;
    if (Doubles[41] > Split) m |= 1ui64 << 41;
    if (Doubles[42] > Split) m |= 1ui64 << 42;
    if (Doubles[43] > Split) m |= 1ui64 << 43;
    if (Doubles[44] > Split) m |= 1ui64 << 44;
    if (Doubles[45] > Split) m |= 1ui64 << 45;
    if (Doubles[46] > Split) m |= 1ui64 << 46;
    if (Doubles[47] > Split) m |= 1ui64 << 47;
    if (Doubles[48] > Split) m |= 1ui64 << 48;
    if (Doubles[49] > Split) m |= 1ui64 << 49;
    if (Doubles[50] > Split) m |= 1ui64 << 50;
    if (Doubles[51] > Split) m |= 1ui64 << 51;
    if (Doubles[52] > Split) m |= 1ui64 << 52;
    if (Doubles[53] > Split) m |= 1ui64 << 53;
    if (Doubles[54] > Split) m |= 1ui64 << 54;
    if (Doubles[55] > Split) m |= 1ui64 << 55;
    if (Doubles[56] > Split) m |= 1ui64 << 56;
    if (Doubles[57] > Split) m |= 1ui64 << 57;
    if (Doubles[58] > Split) m |= 1ui64 << 58;
    if (Doubles[59] > Split) m |= 1ui64 << 59;
    if (Doubles[60] > Split) m |= 1ui64 << 60;
    if (Doubles[61] > Split) m |= 1ui64 << 61;
    if (Doubles[62] > Split) m |= 1ui64 << 62;
    if (Doubles[63] > Split) m |= 1ui64 << 63;
    *Int64 = m;
}
__global__ void copyFloatsToInt32sKernel(float* Floats, uint32_t* Int32s, float Split) {
    copyFloatsToInt32Func(&Floats[blockIdx.x << 5], &Int32s[blockIdx.x], Split);
}
__global__ void copyDoublesToInt32sKernel(double* Floats, uint32_t* Int32s, double Split) {
    copyDoublesToInt32Func(&Floats[blockIdx.x << 5], &Int32s[blockIdx.x], Split);
}
__global__ void copyFloatsToInt64sKernel(float* Floats, uint64_t* Int64s, float Split) {
    copyFloatsToInt64Func(&Floats[blockIdx.x << 5], &Int64s[blockIdx.x], Split);
}
__global__ void copyDoublesToInt64sKernel(double* Floats, uint64_t* Int64s, double Split) {
    copyDoublesToInt64Func(&Floats[blockIdx.x << 5], &Int64s[blockIdx.x], Split);
}
__global__ void copyFloatsToBoolsKernel(float* Floats, bool* Bools, float Split) {
    Bools[blockIdx.x] = Floats[blockIdx.x] > Split;
}
__global__ void copyDoublesToBoolsKernel(float* Doubles, bool* Bools, float Split) {
    Bools[blockIdx.x] = Doubles[blockIdx.x] > Split;
}

__host__ void BrendanCUDA::AI::CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split, bool MemoryOnHost) {
    if (MemoryOnHost) {
        for (size_t i = 0; i < Length; ++i) {
            Bools[i] = Floats[i] > Split;
        }
    }
    else {
        copyFloatsToBoolsKernel<<<Length, 1>>>(Floats, Bools, Split);
    }
}
__host__ void BrendanCUDA::AI::CopyDoublesToBools(float* Doubles, bool* Bools, size_t Length, float Split, bool MemoryOnHost) {
    if (MemoryOnHost) {
        for (size_t i = 0; i < Length; ++i) {
            Bools[i] = Doubles[i] > Split;
        }
    }
    else {
        copyDoublesToBoolsKernel<<<Length, 1>>>(Doubles, Bools, Split);
    }
}
__device__ void BrendanCUDA::AI::CopyFloatsToBools(float* Floats, bool* Bools, size_t Length, float Split) {
    for (size_t i = 0; i < Length; ++i) {
        Bools[i] = Floats[i] > Split;
    }
}
__device__ void BrendanCUDA::AI::CopyDoublesToBools(double* Doubles, bool* Bools, size_t Length, double Split) {
    for (size_t i = 0; i < Length; ++i) {
        Bools[i] = Doubles[i] > Split;
    }
}
__host__ void BrendanCUDA::AI::CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split, bool MemoryOnHost) {
    if (MemoryOnHost) {
        for (size_t i = 0; i < Int32Length; ++i) {
            copyFloatsToInt32Func(&Floats[i << 5], &Int32s[i], Split);
        }
    }
    else {
        copyFloatsToInt32sKernel<<<Int32Length, 1>>>(Floats, Int32s, Split);
    }
}
__host__ void BrendanCUDA::AI::CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split, bool MemoryOnHost) {
    if (MemoryOnHost) {
        for (size_t i = 0; i < Int32Length; ++i) {
            copyDoublesToInt32Func(&Doubles[i << 5], &Int32s[i], Split);
        }
    }
    else {
        copyDoublesToInt32sKernel<<<Int32Length, 1>>>(Doubles, Int32s, Split);
    }
}
__device__ void BrendanCUDA::AI::CopyFloatsToInt32s(float* Floats, uint32_t* Int32s, size_t Int32Length, float Split) {
    for (size_t i = 0; i < Int32Length; ++i) {
        copyFloatsToInt32Func(&Floats[i << 5], &Int32s[i], Split);
    }
}
__device__ void BrendanCUDA::AI::CopyDoublesToInt32s(double* Doubles, uint32_t* Int32s, size_t Int32Length, double Split) {
    for (size_t i = 0; i < Int32Length; ++i) {
        copyDoublesToInt32Func(&Doubles[i << 5], &Int32s[i], Split);
    }
}
__host__ void BrendanCUDA::AI::CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split, bool MemoryOnHost) {
    if (MemoryOnHost) {
        for (size_t i = 0; i < Int64Length; ++i) {
            copyFloatsToInt64Func(&Floats[i << 5], &Int64s[i], Split);
        }
    }
    else {
        copyFloatsToInt64sKernel<<<Int64Length, 1>>>(Floats, Int64s, Split);
    }
}
__host__ void BrendanCUDA::AI::CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split, bool MemoryOnHost) {
    if (MemoryOnHost) {
        for (size_t i = 0; i < Int64Length; ++i) {
            copyDoublesToInt64Func(&Doubles[i << 5], &Int64s[i], Split);
        }
    }
    else {
        copyDoublesToInt64sKernel<<<Int64Length, 1>>>(Doubles, Int64s, Split);
    }
}
__device__ void BrendanCUDA::AI::CopyFloatsToInt64s(float* Floats, uint64_t* Int64s, size_t Int64Length, float Split) {
    for (size_t i = 0; i < Int64Length; ++i) {
        copyFloatsToInt64Func(&Floats[i << 5], &Int64s[i], Split);
    }
}
__device__ void BrendanCUDA::AI::CopyDoublesToInt64s(double* Doubles, uint64_t* Int64s, size_t Int64Length, double Split) {
    for (size_t i = 0; i < Int64Length; ++i) {
        copyDoublesToInt64Func(&Doubles[i << 5], &Int64s[i], Split);
    }
}