#include "brendancuda_nets_net.cuh"
#include <device_launch_parameters.h>

__host__ __device__ BrendanCUDA::Nets::NetNode::NetNode() {
    data = 0;
    inputs = 0;
    inputCount = 0;
    outputs = 0;
    outputCount = 0;
}

__host__ __device__ void BrendanCUDA::Nets::NetNode::Dispose(dataDestructor_t DataDestructor) {
    if (DataDestructor) {
        DataDestructor(data);
    }
#if __CUDA_ARCH__
    delete[] inputs;
    delete[] outputs;
#else
    cudaFree(inputs);
    cudaFree(outputs);
#endif
}

BrendanCUDA::Nets::Net::Net() 
    : nodes(*(new thrust::device_vector<NetNode>())) {}

__global__ void disposeKernel(BrendanCUDA::Nets::NetNode* nodes, BrendanCUDA::Nets::dataDestructor_t DataDestructor) {
    nodes[blockIdx.x].Dispose(DataDestructor);
}
void BrendanCUDA::Nets::Net::Dispose(dataDestructor_t DataDestructor) {
    disposeKernel<<<nodes.size(), 1>>>(nodes.data().get(), DataDestructor);
    delete (&nodes);
}

thrust::device_ptr<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::Data() {
    return nodes.data();
}

thrust::device_reference<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::operator[](size_t i) {
    return nodes[i];
}

void BrendanCUDA::Nets::Net::RemoveAt(size_t Index, dataDestructor_t DataDestructor) {
    NetNode nn = nodes[Index];
    size_t* inputs = new size_t[nn.inputCount];
    size_t* outputs = new size_t[nn.outputCount];
    if (cudaMemcpy(inputs, nn.inputs, sizeof(size_t) * nn.inputCount, cudaMemcpyDeviceToHost)) {
        throw std::exception();
    }
    if (cudaMemcpy(outputs, nn.outputs, sizeof(size_t) * nn.outputCount, cudaMemcpyDeviceToHost)) {
        throw std::exception();
    }
    for (size_t i = 0; i < nn.inputCount; ++i) {
        NetNode nn2 = nodes[inputs[i]];
        size_t* outputs2 = new size_t[nn2.outputCount];
        size_t ocm1 = nn2.outputCount - 1;
        if (cudaMemcpy(outputs2, nn2.outputs, sizeof(size_t) * nn2.outputCount, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        for (size_t j = 0; j < nn2.outputCount; ++j) {
            if (outputs2[j] == Index) {
                outputs2[j] = outputs[ocm1];
                break;
            }
        }
        size_t* outputs3;
        if (cudaMalloc(&outputs3, sizeof(size_t) * ocm1)) {
            throw std::exception();
        }
        if (cudaMemcpy(outputs3, outputs2, sizeof(size_t) * ocm1, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        cudaFree(nn2.outputs);
        nn2.outputs = outputs3;
        nn2.outputCount = ocm1;
        nodes[inputs[i]] = nn2;
        delete[] outputs2;
    }
    for (size_t i = 0; i < nn.outputCount; ++i) {
        NetNode nn2 = nodes[outputs[i]];
        size_t* inputs2 = new size_t[nn2.inputCount];
        size_t icm1 = nn2.inputCount - 1;
        if (cudaMemcpy(inputs2, nn2.inputs, sizeof(size_t) * nn2.inputCount, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        for (size_t j = 0; j < nn2.inputCount; ++j) {
            if (inputs2[j] == Index) {
                inputs2[j] = inputs[icm1];
                break;
            }
        }
        size_t* inputs3;
        if (cudaMalloc(&inputs3, sizeof(size_t) * icm1)) {
            throw std::exception();
        }
        if (cudaMemcpy(inputs3, inputs2, sizeof(size_t) * icm1, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        cudaFree(nn2.inputs);
        nn2.inputs = inputs3;
        nn2.inputCount = icm1;
        nodes[outputs[i]] = nn2;
        delete[] inputs2;
    }
    delete[] inputs;
    delete[] outputs;
    nn.Dispose(DataDestructor);
    nodes[Index] = nodes[nodes.size() - 1];
    nodes.pop_back();
}