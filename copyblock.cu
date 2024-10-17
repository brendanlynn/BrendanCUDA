#include "copyblock.h"

namespace bcuda {
    namespace details {
        __host__ __device__ ArrayV<Landmark> GetLandmarksInDirection(uint32_t InputLength, uint32_t OutputLength, uint32_t RangeLength, uint32_t InputIndex, uint32_t OutputIndex) {
            Landmark* landmarks = 0;
            size_t landmarkSize = 0;
            size_t landmarkCapacity = 0;

            while (RangeLength) {
                uint32_t dInput = InputLength - InputIndex;
                uint32_t dOutput = OutputLength - OutputIndex;
                bool oG = dOutput > dInput;
                uint32_t dMin = oG ? dInput : dOutput;
                if (dMin > RangeLength) dMin = RangeLength;

                if (landmarkCapacity <= landmarkSize + 1) {
                    if (landmarkCapacity == 0) landmarkCapacity = 1;
                    else landmarkCapacity <<= 1;
                    Landmark* newArr = new Landmark[landmarkCapacity];
                    memcpy(newArr, landmarks, landmarkSize * sizeof(Landmark));
                    delete[] landmarks;
                    landmarks = newArr;
                }
                landmarks[landmarkSize++] = Landmark(InputIndex, OutputIndex, dMin);

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
            ArrayV<Landmark> landmarkArray(landmarkSize);
            if (landmarks) memcpy(landmarkArray.Data(), landmarks, sizeof(Landmark) * landmarkSize);
            delete[] landmarks;
            return landmarkArray;
        }
    }
}