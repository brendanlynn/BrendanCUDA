#pragma once

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    namespace Nets {
        using dataDestructor_t = void (*)(void*);

        struct NetNode final {
            void* data;
            NetNode** inputs;
            size_t inputCount;
            NetNode** outputs;
            size_t outputCount;

            __host__ __device__ NetNode();

            __host__ __device__ void Dispose(dataDestructor_t DataDestructor);
        };
        class Net final {
        public:
            Net();
            void Dispose(dataDestructor_t DataDestructor);
            thrust::device_vector<NetNode>& DataVec();
            thrust::device_ptr<NetNode> DataPtr();
            thrust::device_reference<NetNode> operator[](size_t i);

            static bool AddConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
            static bool AddConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
            static bool AddConnection(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
            static bool RemoveConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
            static bool RemoveConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
            static bool RemoveConnection(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
            static void RemoveAllConnections(NetNode* Node);
        private:
            thrust::device_vector<NetNode>& nodes;
        };

    }
}