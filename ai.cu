#include "ai.h"
#include "errorhelp.h"
#include <curand_kernel.h>

template <std::floating_point _TFloat>
__host__ __device__ uint32_t convertFloatsToInt32(_TFloat* Floats, size_t Max, _TFloat Split) {
    uint32_t m = 0;
    if (Max >= 32) {
        if (*(Floats++) > Split) m |= 1ui32 << 0;
        if (*(Floats++) > Split) m |= 1ui32 << 1;
        if (*(Floats++) > Split) m |= 1ui32 << 2;
        if (*(Floats++) > Split) m |= 1ui32 << 3;
        if (*(Floats++) > Split) m |= 1ui32 << 4;
        if (*(Floats++) > Split) m |= 1ui32 << 5;
        if (*(Floats++) > Split) m |= 1ui32 << 6;
        if (*(Floats++) > Split) m |= 1ui32 << 7;
        if (*(Floats++) > Split) m |= 1ui32 << 8;
        if (*(Floats++) > Split) m |= 1ui32 << 9;
        if (*(Floats++) > Split) m |= 1ui32 << 10;
        if (*(Floats++) > Split) m |= 1ui32 << 11;
        if (*(Floats++) > Split) m |= 1ui32 << 12;
        if (*(Floats++) > Split) m |= 1ui32 << 13;
        if (*(Floats++) > Split) m |= 1ui32 << 14;
        if (*(Floats++) > Split) m |= 1ui32 << 15;
        if (*(Floats++) > Split) m |= 1ui32 << 16;
        if (*(Floats++) > Split) m |= 1ui32 << 17;
        if (*(Floats++) > Split) m |= 1ui32 << 18;
        if (*(Floats++) > Split) m |= 1ui32 << 19;
        if (*(Floats++) > Split) m |= 1ui32 << 20;
        if (*(Floats++) > Split) m |= 1ui32 << 21;
        if (*(Floats++) > Split) m |= 1ui32 << 22;
        if (*(Floats++) > Split) m |= 1ui32 << 23;
        if (*(Floats++) > Split) m |= 1ui32 << 24;
        if (*(Floats++) > Split) m |= 1ui32 << 25;
        if (*(Floats++) > Split) m |= 1ui32 << 26;
        if (*(Floats++) > Split) m |= 1ui32 << 27;
        if (*(Floats++) > Split) m |= 1ui32 << 28;
        if (*(Floats++) > Split) m |= 1ui32 << 29;
        if (*(Floats++) > Split) m |= 1ui32 << 30;
        if (*Floats > Split) m |= 1ui32 << 31;
    }
    else {
        Floats += 32;
        switch (Max) {
        case 32: if (*(Floats--) > Split) m |= 1ui32 << 31;
        case 31: if (*(Floats--) > Split) m |= 1ui32 << 30;
        case 30: if (*(Floats--) > Split) m |= 1ui32 << 29;
        case 29: if (*(Floats--) > Split) m |= 1ui32 << 28;
        case 28: if (*(Floats--) > Split) m |= 1ui32 << 27;
        case 27: if (*(Floats--) > Split) m |= 1ui32 << 26;
        case 26: if (*(Floats--) > Split) m |= 1ui32 << 25;
        case 25: if (*(Floats--) > Split) m |= 1ui32 << 24;
        case 24: if (*(Floats--) > Split) m |= 1ui32 << 23;
        case 23: if (*(Floats--) > Split) m |= 1ui32 << 22;
        case 22: if (*(Floats--) > Split) m |= 1ui32 << 21;
        case 21: if (*(Floats--) > Split) m |= 1ui32 << 20;
        case 20: if (*(Floats--) > Split) m |= 1ui32 << 19;
        case 19: if (*(Floats--) > Split) m |= 1ui32 << 18;
        case 18: if (*(Floats--) > Split) m |= 1ui32 << 17;
        case 17: if (*(Floats--) > Split) m |= 1ui32 << 16;
        case 16: if (*(Floats--) > Split) m |= 1ui32 << 15;
        case 15: if (*(Floats--) > Split) m |= 1ui32 << 14;
        case 14: if (*(Floats--) > Split) m |= 1ui32 << 13;
        case 13: if (*(Floats--) > Split) m |= 1ui32 << 12;
        case 12: if (*(Floats--) > Split) m |= 1ui32 << 11;
        case 11: if (*(Floats--) > Split) m |= 1ui32 << 10;
        case 10: if (*(Floats--) > Split) m |= 1ui32 << 9;
        case 9: if (*(Floats--) > Split) m |= 1ui32 << 8;
        case 8: if (*(Floats--) > Split) m |= 1ui32 << 7;
        case 7: if (*(Floats--) > Split) m |= 1ui32 << 6;
        case 6: if (*(Floats--) > Split) m |= 1ui32 << 5;
        case 5: if (*(Floats--) > Split) m |= 1ui32 << 4;
        case 4: if (*(Floats--) > Split) m |= 1ui32 << 3;
        case 3: if (*(Floats--) > Split) m |= 1ui32 << 2;
        case 2: if (*(Floats--) > Split) m |= 1ui32 << 1;
        case 1: if (*(Floats--) > Split) m |= 1ui32 << 0;
        }
    }
    return m;
}
template <std::floating_point _TFloat>
__host__ __device__ void convertInt32ToFloats(_TFloat* Floats, uint32_t Int, size_t Max, _TFloat ValTrue, _TFloat ValFalse) {
    if (Max >= 32) {
        *(Floats++) = (Int & (1ui32 << 0)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 1)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 2)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 3)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 4)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 5)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 6)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 7)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 8)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 9)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 10)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 11)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 12)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 13)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 14)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 15)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 16)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 17)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 18)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 19)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 20)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 21)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 22)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 23)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 24)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 25)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 26)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 27)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 28)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 29)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 30)) ? ValTrue : ValFalse;
        *(Floats++) = (Int & (1ui32 << 31)) ? ValTrue : ValFalse;
    }
    else {
        Floats += 32;
        switch (Max) {
        case 32: *(Floats++) = (Int & (1ui32 << 31)) ? ValTrue : ValFalse;
        case 31: *(Floats++) = (Int & (1ui32 << 30)) ? ValTrue : ValFalse;
        case 30: *(Floats++) = (Int & (1ui32 << 29)) ? ValTrue : ValFalse;
        case 29: *(Floats++) = (Int & (1ui32 << 28)) ? ValTrue : ValFalse;
        case 28: *(Floats++) = (Int & (1ui32 << 27)) ? ValTrue : ValFalse;
        case 27: *(Floats++) = (Int & (1ui32 << 26)) ? ValTrue : ValFalse;
        case 26: *(Floats++) = (Int & (1ui32 << 25)) ? ValTrue : ValFalse;
        case 25: *(Floats++) = (Int & (1ui32 << 24)) ? ValTrue : ValFalse;
        case 24: *(Floats++) = (Int & (1ui32 << 23)) ? ValTrue : ValFalse;
        case 23: *(Floats++) = (Int & (1ui32 << 22)) ? ValTrue : ValFalse;
        case 22: *(Floats++) = (Int & (1ui32 << 21)) ? ValTrue : ValFalse;
        case 21: *(Floats++) = (Int & (1ui32 << 20)) ? ValTrue : ValFalse;
        case 20: *(Floats++) = (Int & (1ui32 << 19)) ? ValTrue : ValFalse;
        case 19: *(Floats++) = (Int & (1ui32 << 18)) ? ValTrue : ValFalse;
        case 18: *(Floats++) = (Int & (1ui32 << 17)) ? ValTrue : ValFalse;
        case 17: *(Floats++) = (Int & (1ui32 << 16)) ? ValTrue : ValFalse;
        case 16: *(Floats++) = (Int & (1ui32 << 15)) ? ValTrue : ValFalse;
        case 15: *(Floats++) = (Int & (1ui32 << 14)) ? ValTrue : ValFalse;
        case 14: *(Floats++) = (Int & (1ui32 << 13)) ? ValTrue : ValFalse;
        case 13: *(Floats++) = (Int & (1ui32 << 12)) ? ValTrue : ValFalse;
        case 12: *(Floats++) = (Int & (1ui32 << 11)) ? ValTrue : ValFalse;
        case 11: *(Floats++) = (Int & (1ui32 << 10)) ? ValTrue : ValFalse;
        case 10: *(Floats++) = (Int & (1ui32 << 9)) ? ValTrue : ValFalse;
        case 9: *(Floats++) = (Int & (1ui32 << 8)) ? ValTrue : ValFalse;
        case 8: *(Floats++) = (Int & (1ui32 << 7)) ? ValTrue : ValFalse;
        case 7: *(Floats++) = (Int & (1ui32 << 6)) ? ValTrue : ValFalse;
        case 6: *(Floats++) = (Int & (1ui32 << 5)) ? ValTrue : ValFalse;
        case 5: *(Floats++) = (Int & (1ui32 << 4)) ? ValTrue : ValFalse;
        case 4: *(Floats++) = (Int & (1ui32 << 3)) ? ValTrue : ValFalse;
        case 3: *(Floats++) = (Int & (1ui32 << 2)) ? ValTrue : ValFalse;
        case 2: *(Floats++) = (Int & (1ui32 << 1)) ? ValTrue : ValFalse;
        case 1: *(Floats++) = (Int & (1ui32 << 0)) ? ValTrue : ValFalse;
        }
    }
}

__global__ void convertFloatsToBoolsKernel(float* Floats, bool* Bools, float Split) {
    Bools[blockIdx.x] = Floats[blockIdx.x] > Split;
}
__global__ void convertFloatsToBoolsKernel(double* Doubles, bool* Bools, double Split) {
    Bools[blockIdx.x] = Doubles[blockIdx.x] > Split;
}
__global__ void convertBoolsToFloatsKernel(bool* Bools, float* Floats, float ValTrue, float ValFalse) {
    Floats[blockIdx.x] = Bools[blockIdx.x] ? ValTrue : ValFalse;
}
__global__ void convertBoolsToFloatsKernel(bool* Bools, double* Doubles, double ValTrue, double ValFalse) {
    Doubles[blockIdx.x] = Bools[blockIdx.x] ? ValTrue : ValFalse;
}
__global__ void convertFloatsToInt32sKernel(float* Floats, uint32_t* Int32s, size_t Max, float Split) {
    Int32s[blockIdx.x] = convertFloatsToInt32(Floats + (blockIdx.x << 5), Max - (blockIdx.x << 5), Split);
}
__global__ void convertFloatsToInt32sKernel(double* Doubles, uint32_t* Int32s, size_t Max, double Split) {
    Int32s[blockIdx.x] = convertFloatsToInt32(Doubles + (blockIdx.x << 5), Max - (blockIdx.x << 5), Split);
}
__global__ void convertInt32sToFloatsKernel(uint32_t* Int32s, float* Floats, size_t Max, float ValTrue, float ValFalse) {
    convertInt32ToFloats(Floats + (blockIdx.x << 5), Int32s[blockIdx.x], Max - (blockIdx.x << 5), ValTrue, ValFalse);
}
__global__ void convertInt32sToFloatsKernel(uint32_t* Int32s, double* Doubles, size_t Max, double ValTrue, double ValFalse) {
    convertInt32ToFloats(Doubles + (blockIdx.x << 5), Int32s[blockIdx.x], Max - (blockIdx.x << 5), ValTrue, ValFalse);
}

template <std::floating_point _TFloat>
__host__ __forceinline void convertFloatsToBools(_TFloat* Floats, bool* Bools, size_t Length, _TFloat Split) {
    convertFloatsToBoolsKernel<<<Length, 1>>>(Floats, Bools, Split);
}
template <std::floating_point _TFloat>
__host__ __forceinline void convertBoolsToFloats(bool* Bools, _TFloat* Floats, size_t Length, _TFloat ValTrue, _TFloat ValFalse) {
    convertBoolsToFloatsKernel<<<Length, 1>>>(Bools, Floats, ValTrue, ValFalse);
}
template <std::floating_point _TFloat>
__host__ __forceinline void convertFloatsToInt32s(_TFloat* Floats, uint32_t* Int32s, size_t Length, _TFloat Split) {
    convertFloatsToInt32sKernel<<<((Length + 31) >> 5), 1>>>(Floats, Int32s, Length, Split);
}
template <std::floating_point _TFloat>
__host__ __forceinline void convertInt32sToFloats(uint32_t* Int32s, _TFloat* Floats, size_t Length, _TFloat ValTrue, _TFloat ValFalse) {
    convertInt32sToFloatsKernel<<<((Length + 31) >> 5), 1>>>(Floats, Int32s, Length, ValTrue, ValFalse);
}

template <bool _FloatsOnHost, std::floating_point _TFloat, bool _BoolsOnHost>
__host__ void bcuda::ai::ConvertFloatsToBools(_TFloat* Floats, bool* Bools, size_t Length, _TFloat Split) {
    if constexpr (_FloatsOnHost) {
        if constexpr (_BoolsOnHost) {
            for (bool* boolsU = Bools + Length; Bools < boolsU; ++Floats, ++Bools)
                *Bools = *Floats > Split;
        }
        else {
            _TFloat* dFloats;
            cudaMalloc(&dFloats, sizeof(_TFloat) * Length);
            cudaMemcpy(dFloats, Floats, sizeof(_TFloat) * Length, cudaMemcpyHostToDevice);
            convertFloatsToBools(dFloats, Bools, Length, Split);
            cudaFree(dFloats);
        }
    }
    else {
        if constexpr (_BoolsOnHost) {
            bool* dBools;
            cudaMalloc(&dBools, sizeof(bool) * Length);
            convertFloatsToBools(Floats, Bools, Length, Split);
            cudaMemcpy(Bools, dBools, sizeof(bool) * Length, cudaMemcpyDeviceToHost);
        }
        else {
            convertFloatsToBools(Floats, Bools, Length, Split);
        }
    }
}
template <std::floating_point _TFloat>
__device__ void bcuda::ai::ConvertFloatsToBools(_TFloat* Floats, bool* Bools, size_t Length, _TFloat Split) {
    for (bool* boolsU = Bools + Length; Bools < boolsU; ++Floats, ++Bools)
        *Bools = *Floats > Split;
}
template <bool _FloatsOnHost, std::floating_point _TFloat, bool _IntsOnHost, std::integral _TInt>
__host__ void bcuda::ai::ConvertFloatsToInts(_TFloat* Floats, _TInt* Ints, size_t FloatsLength, _TFloat Split) {
    if constexpr (_FloatsOnHost) {
        if constexpr (_IntsOnHost) {
            size_t intLength = (FloatsLength + 31) >> 5;
            for (_TInt* intsU = Ints + intLength; Ints < intsU; Floats += 32, ++Ints, FloatsLength -= 32)
                *Ints = convertFloatsToInt32(Floats, FloatsLength, Split);
        }
        else {
            _TFloat* dFloats;
            cudaMalloc(&dFloats, sizeof(_TFloat) * FloatsLength);
            cudaMemcpy(dFloats, Floats, sizeof(_TFloat) * FloatsLength, cudaMemcpyHostToDevice);
            convertFloatsToInt32s(dFloats, Ints, FloatsLength, Split);
            cudaFree(dFloats);
        }
    }
    else {
        if constexpr (_IntsOnHost) {
            size_t intLength = (FloatsLength + 31) >> 5;
            _TInt* dInts;
            cudaMalloc(&dInts, sizeof(_TInt) * intLength);
            convertFloatsToInt32s(Floats, dInts, FloatsLength, Split);
            cudaMemcpy(Ints, dInts, sizeof(_TInt) * intLength, cudaMemcpyDeviceToHost);
        }
        else {
            convertFloatsToInt32s(Floats, Ints, FloatsLength, Split);
        }
    }
}
template <std::floating_point _TFloat, std::integral _TInt>
__device__ void bcuda::ai::ConvertFloatsToInts(_TFloat* Floats, _TInt* Ints, size_t FloatsLength, _TFloat Split) {
    size_t intLength = (FloatsLength + 31) >> 5;
    for (_TInt* intsU = Ints + intLength; Ints < intsU; Floats += 32, ++Ints, FloatsLength -= 32)
        *Ints = convertFloatsToInt32(Floats, FloatsLength, Split);
}
template <bool _FloatsOnHost, bool _BoolsOnHost, std::floating_point _TFloat>
__host__ void bcuda::ai::ConvertBoolsToFloats(bool* Bools, _TFloat* Floats, size_t Length, _TFloat ValFalse, _TFloat ValTrue) {
    if constexpr (_BoolsOnHost) {
        if constexpr (_FloatsOnHost) {
            for (bool* boolsU = Bools + Length; Bools < boolsU; ++Floats, ++Bools)
                *Floats = *Bools ? ValTrue : ValFalse;
        }
        else {
            bool* dBools;
            cudaMalloc(&dBools, sizeof(_TFloat) * Length);
            cudaMemcpy(dBools, Bools, sizeof(_TFloat) * Length, cudaMemcpyHostToDevice);
            convertBoolsToFloats(dBools, Floats, Length, ValTrue, ValFalse);
            cudaFree(dBools);
        }
    }
    else {
        if constexpr (_FloatsOnHost) {
            _TFloat* dFloats;
            cudaMalloc(&dFloats, sizeof(_TFloat) * Length);
            convertBoolsToFloats(Bools, dFloats, Length, ValTrue, ValFalse);
            cudaMemcpy(Floats, dFloats, sizeof(_TFloat) * Length, cudaMemcpyDeviceToHost);
        }
        else {
            convertBoolsToFloats(Bools, Floats, Length, ValTrue, ValFalse);
        }
    }
}
template <std::floating_point _TFloat>
__device__ void bcuda::ai::ConvertBoolsToFloats(bool* Bools, _TFloat* Floats, size_t Length, _TFloat ValFalse, _TFloat ValTrue) {
    for (bool* boolsU = Bools + Length; Bools < boolsU; ++Floats, ++Bools)
        *Floats = *Bools ? ValTrue : ValFalse;
}
template <bool _IntsOnHost, std::integral _TInt, bool _FloatsOnHost, std::floating_point _TFloat>
__host__ void bcuda::ai::ConvertIntsToFloats(_TInt* Ints, _TFloat* Floats, size_t FloatsLength, _TFloat ValFalse, _TFloat ValTrue) {
    if constexpr (_IntsOnHost) {
        if constexpr (_FloatsOnHost) {
            size_t intLength = (FloatsLength + 31) >> 5;
            for (_TInt* intsU = Ints + intLength; Ints < intsU; Floats += 32, ++Ints, FloatsLength -= 32)
                convertInt32ToFloats(Ints, Floats, FloatsLength, ValTrue, ValFalse);
        }
        else {
            size_t intLength = (FloatsLength + 31) >> 5;
            _TInt* dInts;
            cudaMalloc(&dInts, sizeof(_TInt) * intLength);
            cudaMemcpy(dInts, Ints, sizeof(_TInt) * intLength, cudaMemcpyHostToDevice);
            convertInt32sToFloats(dInts, Floats, FloatsLength, ValTrue, ValFalse);
            cudaFree(dInts);
        }
    }
    else {
        if constexpr (_FloatsOnHost) {
            _TFloat* dFloats;
            cudaMalloc(&dFloats, sizeof(_TFloat) * FloatsLength);
            convertInt32sToFloats(Ints, dFloats, FloatsLength, ValTrue, ValFalse);
            cudaMemcpy(Floats, dFloats, sizeof(_TFloat) * FloatsLength, cudaMemcpyDeviceToHost);
        }
        else {
            convertInt32sToFloats(Ints, Floats, FloatsLength, ValTrue, ValFalse);
        }
    }
}
template <std::integral _TInt, std::floating_point _TFloat>
__device__ void bcuda::ai::ConvertIntsToFloats(_TInt* Ints, _TFloat* Floats, size_t FloatsLength, _TFloat ValFalse, _TFloat ValTrue) {
    size_t intLength = (FloatsLength + 31) >> 5;
    for (_TInt* intsU = Ints + intLength; Ints < intsU; Floats += 32, ++Ints, FloatsLength -= 32)
        convertInt32ToFloats(Ints, Floats, FloatsLength, ValTrue, ValFalse);
}