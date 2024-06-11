#include "brendancuda_nets_net.cuh"
#include <device_launch_parameters.h>

__host__ __device__ BrendanCUDA::Nets::NetNode::NetNode() {
    data = 0;
    inputs = 0;
    inputCount = 0;
    outputs = 0;
    outputCount = 0;
}

__host__ __device__ void BrendanCUDA::Nets::NetNode::Destroy(dataDestructor_t DataDestructor) {
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

__global__ void destroyKernel(BrendanCUDA::Nets::NetNode* nodes, BrendanCUDA::Nets::dataDestructor_t DataDestructor) {
    nodes[blockIdx.x].Destroy(DataDestructor);
}
void BrendanCUDA::Nets::Net::Destroy(dataDestructor_t DataDestructor) {
    destroyKernel<<<nodes.size(), 1>>>(nodes.data().get(), DataDestructor);
    delete (&nodes);
}

thrust::device_ptr<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::Data() {
    return nodes.data();
}

thrust::device_reference<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::operator[](size_t i) {
    return nodes[i];
}