#include "brendancuda_nets_net.cuh"

BrendanCUDA::Nets::NetNode::NetNode() {
    data = 0;
    inputs = 0;
    inputCount = 0;
    outputs = 0;
    outputCount = 0;
}

BrendanCUDA::Nets::Net::Net() 
    : nodes(*(new thrust::device_vector<NetNode>())) {}

void BrendanCUDA::Nets::Net::Destroy() {
    delete (&nodes);
}
thrust::device_ptr<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::Data() {
    return nodes.data();
}
thrust::device_reference<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::operator[](size_t i) {
    return nodes[i];
}