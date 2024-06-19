#include "brendancuda_nets_makenet.h"
#include "brendancuda_random_devicerandom.cuh"
#include "brendancuda_random_sseed.cuh"
#include "brendancuda_cudaerrorhelpers.h"
#include "brendancuda_points.cuh"

using BrendanCUDA::float_3;
using BrendanCUDA::Random::DeviceRandom;
using BrendanCUDA::Random::GetSeedOnKernel;
using BrendanCUDA::Coordinates32_3ToIndex64_RM;
using BrendanCUDA::uint32_3;
using BrendanCUDA::ThrowIfBad;
using BrendanCUDA::Nets::NetNode;
using BrendanCUDA::Nets::Net;
using std::make_tuple;
using thrust::device_vector;

struct Bucket {
    size_t* data;
    size_t size;
    size_t capacity;
};

__global__ void fillRandomly(BrendanCUDA::float_3* arr, uint64_t bS) {
    DeviceRandom dr(bS);
    arr[blockIdx.x] = float_3(dr.GetF(), dr.GetF(), dr.GetF());
}

__device__ void addToBucket(Bucket& bucket, size_t value) {
    if (bucket.data) {
        if (bucket.size == bucket.capacity) {
            bucket.capacity <<= 1;
            size_t* nd = new size_t[bucket.capacity];
            memcpy(nd, bucket.data, sizeof(size_t) * bucket.size);
        }
        bucket.data[bucket.size] = value;
        ++bucket.size;
    }
    else {
        bucket.data = new size_t{ value };
        bucket.size = 1;
        bucket.capacity = 1;
    }
}

__global__ void fillBuckets1(Bucket* bucketData, size_t bucketCountPerD, BrendanCUDA::float_3* data, size_t dataCount) {
    const uint32_3 myBucket(blockIdx.x, blockIdx.y, blockIdx.z);

    size_t i = Coordinates32_3ToIndex64_RM(uint32_3(bucketCountPerD, bucketCountPerD, bucketCountPerD), myBucket);
    Bucket& bucket = bucketData[i];

    bucket = Bucket{0, 0, 0};

    float fv = 1.f / bucketCountPerD;
    float x_lb = fv * myBucket.x;
    float x_ub = fv * (myBucket.x + 1);
    float y_lb = fv * myBucket.y;
    float y_ub = fv * (myBucket.y + 1);
    float z_lb = fv * myBucket.z;
    float z_ub = fv * (myBucket.z + 1);

    for (size_t i = 0; i < dataCount; ++i) {
        float_3 p = data[i];
        if ((p.x > x_lb && p.x <= x_ub && p.y > y_lb && p.y <= y_ub && p.z > z_lb && p.z <= z_ub) || ((!myBucket.x) && (p.x == 0.f)) || ((!myBucket.y) && (p.y == 0.f)) || ((!myBucket.z) && (p.z == 0.f))) {
            addToBucket(bucket, i);
        }
    }
}

__global__ void fillBuckets2(Bucket* nodeData, Bucket* bucketData, uint32_t bucketCountPerD, BrendanCUDA::float_3* data, float ConnectionRange) {
    float_3 p = data[blockIdx.x];
    Bucket& mnd = nodeData[blockIdx.x];

    uint32_t bX = (uint32_t)std::floor(p.x * (float)bucketCountPerD);
    uint32_t bY = (uint32_t)std::floor(p.y * (float)bucketCountPerD);
    uint32_t bZ = (uint32_t)std::floor(p.z * (float)bucketCountPerD);
    uint32_t bC = (uint32_t)std::ceil(ConnectionRange * (float)bucketCountPerD);

    uint32_t bucketCountPerDM1 = bucketCountPerD - 1;
    uint32_t lX = std::max(bX - bC, 0ui32);
    uint32_t uX = std::min(bX + bC, bucketCountPerDM1);
    uint32_t lY = std::max(bY - bC, 0ui32);
    uint32_t uY = std::min(bY + bC, bucketCountPerDM1);
    uint32_t lZ = std::max(bZ - bC, 0ui32);
    uint32_t uZ = std::min(bZ + bC, bucketCountPerDM1);

    float cr_sq = ConnectionRange * ConnectionRange;
    mnd = { 0, 0, 0 };
    for (uint32_t x = lX; x <= uX; ++x) {
        for (uint32_t y = lY; y <= uY; ++y) {
            for (uint32_t z = lZ; z <= uZ; ++z) {
                size_t i = Coordinates32_3ToIndex64_RM(uint32_3(bucketCountPerD, bucketCountPerD, bucketCountPerD), uint32_3(x, y, z));
                float_3 tp = data[i];

                if (cr_sq < (p - tp).MagnatudeSquared()) {
                    addToBucket(mnd, i);
                }
            }
        }
    }
}

__global__ void fillNetNodes(NetNode* netNodes, Bucket* nodeDataBuckets) {
    NetNode& nn = netNodes[blockIdx.x];
    Bucket b = nodeDataBuckets[blockIdx.x];

    nn.data = 0;
    
    nn.inputCount = b.size;
    nn.outputCount = b.size;
    nn.inputs = new NetNode*[b.size];
    nn.outputs = new NetNode*[b.size];

    for (size_t i = 0; i < b.size; ++i) {
        nn.inputs[i] = netNodes + b.data[i];
        nn.outputs[i] = netNodes + b.data[i];
    }
}

__global__ void disposeOfBuckets(Bucket* buckets) {
    delete[] buckets[blockIdx.x].data;
}

BrendanCUDA::Nets::Net BrendanCUDA::Nets::MakeNet_3D(size_t NodeCount, float ConnectionRange, BrendanCUDA::Random::AnyRNG<uint64_t> RNG, thrust::device_vector<float_3>** NodePoints) {
    thrust::device_vector<float_3> dv(NodeCount);

    fillRandomly<<<dv.size(), 1>>>(dv.data().get(), RNG());

    constexpr size_t bucketCountPerD = 10;

    Bucket* bucketData;
    ThrowIfBad(cudaMalloc(&bucketData, bucketCountPerD * bucketCountPerD * bucketCountPerD * sizeof(Bucket)));

    fillBuckets1<<<dim3(bucketCountPerD, bucketCountPerD, bucketCountPerD), 1>>>(bucketData, bucketCountPerD, dv.data().get(), dv.size());

    Bucket* nodesData;
    ThrowIfBad(cudaMalloc(&nodesData, NodeCount * sizeof(Bucket)));

    fillBuckets2<<<NodeCount, 1>>>(nodesData, bucketData, bucketCountPerD, dv.data().get(), ConnectionRange);

    device_vector<NetNode>& ndv = *new device_vector<NetNode>(NodeCount);

    fillNetNodes<<<NodeCount, 1>>>(ndv.data().get(), nodesData);

    disposeOfBuckets<<<bucketCountPerD * bucketCountPerD * bucketCountPerD, 1>>>(bucketData);
    disposeOfBuckets<<<NodeCount, 1>>>(bucketData);

    return Net(ndv);
}