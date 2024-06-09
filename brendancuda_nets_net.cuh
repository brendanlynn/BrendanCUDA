#pragma once

#include <thrust/device_vector.h>

namespace BrendanCUDA {
    namespace Nets {
        template <typename TData>
        struct NetNode final {
            TData data;
            size_t* inputs;
            size_t* outputs;
        };
        template <typename TData>
        class Net final {
        public:
            Net();
            NetNode<TData>* Data();
            NetNode<TData> operator[](size_t i);
            NetNode<TData>* Node(size_t i);
            void Destroy();
        private:
            thrust::device_vector<NetNode<TData>>& nodes;
        };
    }
}

template <typename TData>
BrendanCUDA::Nets::Net<TData>::Net() {
    nodes = *(new thrust::device_vector<NetNode<TData>>());
}

template <typename TData>
BrendanCUDA::Nets::NetNode<TData>* BrendanCUDA::Nets::Net<TData>::Data() {
    return nodes.data();
}

template <typename TData>
BrendanCUDA::Nets::NetNode<TData> BrendanCUDA::Nets::Net<TData>::operator[](size_t i) {
    return nodes[i];
}

template <typename TData>
BrendanCUDA::Nets::NetNode<TData>* BrendanCUDA::Nets::Net<TData>::Node(size_t i) {
    return Data() + i;
}

template <typename TData>
void BrendanCUDA::Nets::Net<TData>::Destroy() {
    delete (&nodes);
}