#pragma once

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace Nets {
        using dataDestructor_t = void (*)(void*);
        struct NetNode final {
            void* data;
            size_t* inputs;
            size_t inputCount;
            size_t* outputs;
            size_t outputCount;

            __host__ __device__ NetNode();

            __host__ __device__ void Dispose(dataDestructor_t DataDestructor);
        };
        class Net final {
        public:
            Net();
            void Dispose(dataDestructor_t DataDestructor);
            thrust::device_ptr<NetNode> Data();
            thrust::device_reference<NetNode> operator[](size_t i);
            bool AddConnection_OnlyInput(size_t InputIndex, size_t OutputIndex, bool CheckForPreexistence, bool CheckForAvailableExcess);
            bool AddConnection_OnlyOutput(size_t InputIndex, size_t OutputIndex, bool CheckForPreexistence, bool CheckForAvailableExcess);
            bool AddConnection(size_t InputIndex, size_t OutputIndex, bool CheckForPreexistence, bool CheckForAvailableExcess);
            bool RemoveConnection_OnlyInput(size_t InputIndex, size_t OutputIndex, bool RemoveExcess);
            bool RemoveConnection_OnlyOutput(size_t InputIndex, size_t OutputIndex, bool RemoveExcess);
            bool RemoveConnection(size_t InputIndex, size_t OutputIndex, bool RemoveExcess);
            void RemoveAt(size_t Index, dataDestructor_t DataDestructor);
        private:
            thrust::device_vector<NetNode>& nodes;
        };
    }
}