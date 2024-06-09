#pragma once

#include <thrust/device_vector.h>

namespace BrendanCUDA {
    namespace Nets {
        struct NetNode final {
            void* data;
            size_t* inputs;
            size_t inputCount;
            size_t* outputs;
            size_t outputCount;

            NetNode();
        };
        class Net final {
        public:
            Net();
            void Destroy();
            thrust::device_ptr<NetNode> Data();
            thrust::device_reference<NetNode> operator[](size_t i);
        private:
            thrust::device_vector<NetNode>& nodes;
        };
    }
}