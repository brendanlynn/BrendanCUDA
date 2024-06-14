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
            thrust::device_ptr<NetNode> Data();
            thrust::device_reference<NetNode> operator[](size_t i);
        private:
            thrust::device_vector<NetNode>& nodes;
        };

        bool Net_AddConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
        bool Net_AddConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
        bool Net_AddConnection(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
        bool Net_RemoveConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
        bool Net_RemoveConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
        bool Net_RemoveConnection(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
        void Net_RemoveAllConnections(NetNode* Node);
    }
}