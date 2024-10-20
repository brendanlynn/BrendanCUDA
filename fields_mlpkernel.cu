#include "fields_mlpkernel.h"

__device__ inline constexpr size_t incrementRoll(size_t Coord, size_t DimensionLength) {
    return ++Coord < DimensionLength ? Coord : 0;
}
__device__ inline constexpr size_t decrementRoll(size_t Coord, size_t DimensionLength) {
    return Coord ? Coord - 1 : DimensionLength - 1;
}

__device__ void runMlplFromDeviceNoTemplateF(void* FixedMLPL, float* Inputs, float* Outputs, size_t InputCount, size_t OutputCount, bcuda::ai::activationFunction_t<float> ActivationFunc) {
    float* weights = (float*)FixedMLPL;
    float* bias = weights + InputCount * OutputCount;
    for (size_t j = 0; j < OutputCount; ++j) {
        float v = bias[j];
        for (size_t i = 0; i < InputCount; ++i)
            v += weights[j * InputCount + i] * Inputs[i];
        Outputs[j] = ActivationFunc(v);
    }
}
__device__ void runMlpFromDeviceNoTemplateF(void* FixedMLP, float* Inputs, float* Outputs, size_t InputCount, bcuda::Span<const size_t> HiddenWidths, size_t OutputCount, bcuda::ai::activationFunction_t<float> ActivationFunc) {
    float* current = new float[HiddenWidths[0]];

    runMlplFromDeviceNoTemplateF(FixedMLP, Inputs, current, InputCount, HiddenWidths[0], ActivationFunc);

    float* fixedMlplInc = (float*)FixedMLP + (InputCount + 1) * HiddenWidths[0];
    for (size_t i = 1; i < HiddenWidths.size - 1; ++i) {
        size_t inputCount = HiddenWidths[i - 1];
        size_t outputCount = HiddenWidths[i];
        float* newCurrent = new float[HiddenWidths[i]];

        runMlplFromDeviceNoTemplateF(fixedMlplInc, current, newCurrent, inputCount, outputCount, ActivationFunc);

        delete[] current;
        current = newCurrent;

        fixedMlplInc += (inputCount + 1) * outputCount;
    }

    runMlplFromDeviceNoTemplateF(fixedMlplInc, current, Outputs, InputCount, OutputCount, ActivationFunc);
}

__global__ void iteratePredictorFieldKernelF(bcuda::Span<const size_t> Dimensions, void* FieldFData, void* FieldBData, size_t FieldElementSize, void* KernelMLP, bcuda::Span<const size_t> HiddenWidths, bcuda::ai::activationFunction_t<float> ActivationFunc, ptrdiff_t InputThisDiff, size_t InputThisCount, ptrdiff_t InputSharedDiff, size_t InputSharedCount, ptrdiff_t OutputDiff, size_t OutputCount) {
    uint32_t idx = blockIdx.x;
    size_t* pos = new size_t[Dimensions.size];
    for (size_t i = 0; i < Dimensions.size; ++i) {
        pos[i] = idx % Dimensions[i];
        idx /= Dimensions[i];
    }

    auto getValIdx = [Dimensions](size_t* Coords) -> size_t {
        size_t idx = 0;
        for (size_t i = 0; i < Dimensions.size; ++i) {
            idx *= Dimensions[i];
            idx += Coords[i];
        }
        return idx;
    };
    auto idxToElementPtr = [FieldElementSize](void* Base, size_t Idx) -> void* {
        return (char*)Base + Idx * FieldElementSize;
    };

    size_t thisValsIdx = getValIdx(pos);

    size_t inputCount = InputThisCount + InputSharedCount * (bcuda::fields::details::AreaCountByDimension(Dimensions.size) - 1);
    float* inputs = new float[inputCount];
    
    {
        float* inputsWH = inputs;

        {
            void* inputsFromPast = (char*)idxToElementPtr(FieldFData, thisValsIdx) + InputThisDiff;

            memcpy(inputsWH, inputsFromPast, sizeof(float) * InputThisCount);
            inputsWH += InputThisCount;
        }

        size_t* coords = new size_t[Dimensions.size];
        memcpy(coords, pos, sizeof(size_t) * Dimensions.size);

        int32_t* offsets = new int32_t[Dimensions.size];
        offsets[0] = 1;
        for (size_t i = 1; i < Dimensions.size; ++i)
            offsets[i] = 0;

        coords[0] = decrementRoll(pos[0], Dimensions[0]);
        while (true) {
            size_t valsIdx = getValIdx(coords);
            void* neighborsInputs = (char*)idxToElementPtr(FieldFData, valsIdx) + InputSharedDiff;

            memcpy(inputsWH, neighborsInputs, sizeof(float) * InputSharedCount);
            inputsWH += InputSharedCount;

            for (size_t i = 0; i < Dimensions.size; ++i) {
                if (++offsets[i] > 2) {
                    offsets[i] = 0;
                    coords[i] = pos[i];
                    goto ContinueWhile;
                }
                if (offsets[i] == 1) coords[i] = decrementRoll(pos[i], Dimensions[i]);
                else coords[i] = incrementRoll(pos[i], Dimensions[i]);
            }
            break;
        ContinueWhile:
            continue;
        }

        delete[] coords;
        delete[] offsets;
    }

    void* outputs = (float*)((char*)idxToElementPtr(FieldBData, thisValsIdx) + OutputDiff);

    runMlpFromDeviceNoTemplateF(KernelMLP, inputs, (float*)outputs, inputCount, HiddenWidths, OutputCount, ActivationFunc);

    delete[] pos;
    delete[] inputs;
}

__device__ void runMlplFromDeviceNoTemplateD(void* FixedMLPL, double* Inputs, double* Outputs, size_t InputCount, size_t OutputCount, bcuda::ai::activationFunction_t<double> ActivationFunc) {
    double* weights = (double*)FixedMLPL;
    double* bias = weights + InputCount * OutputCount;
    for (size_t j = 0; j < OutputCount; ++j) {
        double v = bias[j];
        for (size_t i = 0; i < InputCount; ++i)
            v += weights[j * InputCount + i] * Inputs[i];
        Outputs[j] = ActivationFunc(v);
    }
}
__device__ void runMlpFromDeviceNoTemplateD(void* FixedMLP, double* Inputs, double* Outputs, size_t InputCount, bcuda::Span<const size_t> HiddenWidths, size_t OutputCount, bcuda::ai::activationFunction_t<double> ActivationFunc) {
    double* current = new double[HiddenWidths[0]];

    runMlplFromDeviceNoTemplateD(FixedMLP, Inputs, current, InputCount, HiddenWidths[0], ActivationFunc);

    double* fixedMlplInc = (double*)FixedMLP + (InputCount + 1) * HiddenWidths[0];
    for (size_t i = 1; i < HiddenWidths.size - 1; ++i) {
        size_t inputCount = HiddenWidths[i - 1];
        size_t outputCount = HiddenWidths[i];
        double* newCurrent = new double[HiddenWidths[i]];

        runMlplFromDeviceNoTemplateD(fixedMlplInc, current, newCurrent, inputCount, outputCount, ActivationFunc);

        delete[] current;
        current = newCurrent;

        fixedMlplInc += (inputCount + 1) * outputCount;
    }

    runMlplFromDeviceNoTemplateD(fixedMlplInc, current, Outputs, InputCount, OutputCount, ActivationFunc);
}

__global__ void iteratePredictorFieldKernelD(bcuda::Span<const size_t> Dimensions, void* FieldFData, void* FieldBData, size_t FieldElementSize, void* KernelMLP, bcuda::Span<const size_t> HiddenWidths, bcuda::ai::activationFunction_t<double> ActivationFunc, ptrdiff_t InputThisDiff, size_t InputThisCount, ptrdiff_t InputSharedDiff, size_t InputSharedCount, ptrdiff_t OutputDiff, size_t OutputCount) {
    uint32_t idx = blockIdx.x;
    size_t* pos = new size_t[Dimensions.size];
    for (size_t i = 0; i < Dimensions.size; ++i) {
        pos[i] = idx % Dimensions[i];
        idx /= Dimensions[i];
    }

    auto getValIdx = [Dimensions](size_t* Coords) -> size_t {
        size_t idx = 0;
        for (size_t i = 0; i < Dimensions.size; ++i) {
            idx *= Dimensions[i];
            idx += Coords[i];
        }
        return idx;
        };
    auto idxToElementPtr = [FieldElementSize](void* Base, size_t Idx) -> void* {
        return (char*)Base + Idx * FieldElementSize;
        };

    size_t thisValsIdx = getValIdx(pos);

    size_t inputCount = InputThisCount + InputSharedCount * (bcuda::fields::details::AreaCountByDimension(Dimensions.size) - 1);
    double* inputs = new double[inputCount];

    {
        double* inputsWH = inputs;

        {
            void* inputsFromPast = (char*)idxToElementPtr(FieldFData, thisValsIdx) + InputThisDiff;

            memcpy(inputsWH, inputsFromPast, sizeof(double) * InputThisCount);
            inputsWH += InputThisCount;
        }

        size_t* coords = new size_t[Dimensions.size];
        memcpy(coords, pos, sizeof(size_t) * Dimensions.size);

        int32_t* offsets = new int32_t[Dimensions.size];
        offsets[0] = 1;
        for (size_t i = 1; i < Dimensions.size; ++i)
            offsets[i] = 0;

        coords[0] = decrementRoll(pos[0], Dimensions[0]);
        while (true) {
            size_t valsIdx = getValIdx(coords);
            void* neighborsInputs = (char*)idxToElementPtr(FieldFData, valsIdx) + InputSharedDiff;

            memcpy(inputsWH, neighborsInputs, sizeof(double) * InputSharedCount);
            inputsWH += InputSharedCount;

            for (size_t i = 0; i < Dimensions.size; ++i) {
                if (++offsets[i] > 2) {
                    offsets[i] = 0;
                    coords[i] = pos[i];
                    goto ContinueWhile;
                }
                if (offsets[i] == 1) coords[i] = decrementRoll(pos[i], Dimensions[i]);
                else coords[i] = incrementRoll(pos[i], Dimensions[i]);
            }
            break;
        ContinueWhile:
            continue;
        }

        delete[] coords;
        delete[] offsets;
    }

    void* outputs = (double*)((char*)idxToElementPtr(FieldBData, thisValsIdx) + OutputDiff);

    runMlpFromDeviceNoTemplateD(KernelMLP, inputs, (double*)outputs, inputCount, HiddenWidths, OutputCount, ActivationFunc);

    delete[] pos;
    delete[] inputs;
}

namespace bcuda {
    namespace fields {
        namespace details {
            template <std::floating_point _TFloat>
            void RunKernelMLPOverDField(Span<const size_t> Dimensions, void* FieldFData, void* FieldBData, size_t FieldElementSize, void* KernelMLP, Span<const size_t> HiddenWidths, ai::activationFunction_t<float> ActivationFunc, ptrdiff_t InputThisDiff, size_t InputThisCount, ptrdiff_t InputSharedDiff, size_t InputSharedCount, ptrdiff_t OutputDiff, size_t OutputCount) {
                size_t* dDimsPtr;
                bcuda::ThrowIfBad(cudaMalloc(&dDimsPtr, sizeof(size_t) * Dimensions.size));
                bcuda::ThrowIfBad(cudaMemcpy(dDimsPtr, Dimensions.ptr, sizeof(size_t) * Dimensions.size, cudaMemcpyHostToDevice));
                bcuda::Span<size_t> dDims(dDimsPtr, Dimensions.size);

                size_t* dHiddenPtr;
                bcuda::ThrowIfBad(cudaMalloc(&dHiddenPtr, sizeof(size_t) * HiddenWidths.size));
                bcuda::ThrowIfBad(cudaMemcpy(dHiddenPtr, HiddenWidths.ptr, sizeof(size_t) * HiddenWidths.size, cudaMemcpyHostToDevice));
                bcuda::Span<size_t> dHiddenWidths(dHiddenPtr, HiddenWidths.size);

                size_t total = 1;
                for (size_t i = 0; i < Dimensions.size; ++i)
                    total *= Dimensions[i];

                if constexpr (std::same_as<_TFloat, float>)
                    iteratePredictorFieldKernelF<<<total, 1>>>(dDims, FieldFData, FieldBData, FieldElementSize, KernelMLP, dHiddenWidths, ActivationFunc, InputThisDiff, InputThisCount, InputSharedDiff, InputSharedCount, OutputDiff, OutputCount);
                else {
                    static_assert(std::same_as<_TFloat, double>, "_TFloat must be float or double.");
                    iteratePredictorFieldKernelD<<<total, 1>>>(dDims, FieldFData, FieldBData, FieldElementSize, KernelMLP, dHiddenWidths, ActivationFunc, InputThisDiff, InputThisCount, InputSharedDiff, InputSharedCount, OutputDiff, OutputCount);
                }

                bcuda::ThrowIfBad(cudaFree(dDimsPtr));
                bcuda::ThrowIfBad(cudaFree(dHiddenPtr));
            }
        }
    }
}