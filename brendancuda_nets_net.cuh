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

            __host__ __device__ void Destroy(dataDestructor_t DataDestructor);
        };
        class Net final {
        public:
            Net();
            void Destroy(dataDestructor_t DataDestructor);
            thrust::device_ptr<NetNode> Data();
            thrust::device_reference<NetNode> operator[](size_t i);
        private:
            thrust::device_vector<NetNode>& nodes;
        };
    }
}