#include "copyblock.h"
#include <vector>

__host__ __device__ bcuda::ArrayV<bcuda::details::Landmark> bcuda::details::GetLandmarksInDirection(uint32_t InputLength, uint32_t OutputLength, uint32_t RangeLength, uint32_t InputIndex, uint32_t OutputIndex) {
    std::vector<Landmark> landmarks;
    while (RangeLength) {
        uint32_t dInput = InputLength - InputIndex;
        uint32_t dOutput = OutputLength - OutputIndex;
        bool oG = dOutput > dInput;
        uint32_t dMin = oG ? dInput : dOutput;
        if (dMin >= RangeLength) break;
        landmarks.push_back(Landmark(InputIndex, OutputIndex, dMin));
        if (oG) {
            InputIndex = 0;
            OutputIndex += dMin;
        }
        else if (dInput == dOutput) {
            InputIndex = 0;
            OutputIndex = 0;
        }
        else {
            InputIndex += dMin;
            OutputIndex = 0;
        }
        RangeLength -= dMin;
    }
    ArrayV<Landmark> landmarkArray(landmarks.size());
    std::copy(landmarks.data(), landmarks.data() + landmarks.size(), landmarkArray.Data());
    return landmarkArray;
}