#include "brendancuda_nets_makenet.h"
#include "brendancuda_rand_drandom.cuh"
#include "brendancuda_rand_sseed.cuh"
#include "brendancuda_errorhelp.h"
#include "brendancuda_points.h"
#include <cuda_runtime.h>

using BrendanCUDA::float_3;
using BrendanCUDA::Random::DeviceRandom;
using BrendanCUDA::Random::GetSeedOnKernel;
using BrendanCUDA::CoordinatesToIndex;
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

struct BucketTS {
    size_t* data;
    size_t size;
    size_t capacity;
    int lock;
};

__device__ uint32_3 whichBucket(float_3 point, uint32_t bucketCountPerD) {
    uint32_t bX = (uint32_t)std::floor(point.x * bucketCountPerD);
    uint32_t bY = (uint32_t)std::floor(point.y * bucketCountPerD);
    uint32_t bZ = (uint32_t)std::floor(point.z * bucketCountPerD);

    if (bX >= bucketCountPerD) {
        bX = bucketCountPerD - 1;
    }
    if (bY >= bucketCountPerD) {
        bY = bucketCountPerD - 1;
    }
    if (bZ >= bucketCountPerD) {
        bZ = bucketCountPerD - 1;
    }

    return uint32_3(bX, bY, bZ);
}

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
            delete[] bucket.data;
            bucket.data = nd;
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

__global__ void initBuckets(BucketTS* buckets) {
    buckets[blockIdx.x] = { 0, 0, 0, 0 };
}

__global__ void initBuckets(Bucket* buckets) {
    buckets[blockIdx.x] = { 0, 0, 0 };
}

__device__ void resizeBucket(size_t*& bucket, size_t newCapacity, size_t size, size_t currentCapacity) {
    if (size > currentCapacity) {
        size = currentCapacity;
    }
    size_t* newArr = new size_t[newCapacity];
    if (bucket) {
        for (size_t i = 0; i < size; ++i) {
            newArr[i] = bucket[i];
        }
        delete[] bucket;
    }
    bucket = newArr;
}

__device__ void addToBucketTS(BucketTS& bucket, size_t value) {
    size_t pos = atomicAdd(&bucket.size, 1);

    while (atomicCAS(&bucket.lock, 0, 1)) {}

    if (pos >= bucket.capacity) {
        size_t nCap = bucket.capacity;
        if (nCap < bucket.size) {
            if (nCap) {
                nCap <<= 1;
            }
            else {
                nCap = 1;
            }
        }
        while (nCap < bucket.size) {
            nCap <<= 1;
        }

        resizeBucket(bucket.data, nCap, pos, bucket.capacity);
        bucket.capacity = nCap;
    }

    bucket.data[pos] = value;

    __threadfence();
    atomicExch(&bucket.lock, 0);
}

__global__ void fillBuckets1(BucketTS* bucketData, size_t bucketCountPerD, BrendanCUDA::float_3* data, size_t dataCount) {
    float_3 p = data[blockIdx.x];
    
    uint32_3 bCs = whichBucket(p, bucketCountPerD);

    size_t bI = CoordinatesToIndex<uint64_t, uint32_t, 3, true>(uint32_3(bucketCountPerD, bucketCountPerD, bucketCountPerD), bCs);

    addToBucketTS(bucketData[bI], blockIdx.x);
}

__global__ void fillBuckets2(Bucket* nodeData, BucketTS* bucketData, uint32_t bucketCountPerD, BrendanCUDA::float_3* data, float ConnectionRange) {
    float_3 p = data[blockIdx.x];
    Bucket& mnd = nodeData[blockIdx.x];

    uint32_3 bCs = whichBucket(p, bucketCountPerD);
    uint32_t bX = bCs.x;
    uint32_t bY = bCs.y;
    uint32_t bZ = bCs.z;
    uint32_t bC = (uint32_t)std::ceil(ConnectionRange * (float)bucketCountPerD);

    uint32_t bucketCountPerDM1 = bucketCountPerD - 1;
    uint32_t lX = bC > bX ? 0 : bX - bC;
    uint32_t uX = bX + bC;
    uint32_t lY = bC > bY ? 0 : bY - bC;
    uint32_t uY = bY + bC;
    uint32_t lZ = bC > bZ ? 0 : bZ - bC;
    uint32_t uZ = bZ + bC;

    if (uX > bucketCountPerDM1) uX = bucketCountPerDM1;
    if (uY > bucketCountPerDM1) uY = bucketCountPerDM1;
    if (uZ > bucketCountPerDM1) uZ = bucketCountPerDM1;

    float cr_sq = ConnectionRange * ConnectionRange;
    mnd = { 0, 0, 0 };
    for (uint32_t x = lX; x <= uX; ++x) {
        for (uint32_t y = lY; y <= uY; ++y) {
            for (uint32_t z = lZ; z <= uZ; ++z) {
                size_t bucketIndex = CoordinatesToIndex<uint64_t, uint32_t, 3, true>(uint32_3(bucketCountPerD, bucketCountPerD, bucketCountPerD), uint32_3(x, y, z));
                BucketTS bucket = bucketData[bucketIndex];
                for (size_t i = 0; i < bucket.size; ++i) {
                    size_t j = bucket.data[i];
                    float_3 tp = data[j];

                    if (cr_sq > (p - tp).MagnatudeSquared()) {
                        addToBucket(mnd, j);
                    }
                }
            }
        }
    }
}

__global__ void fillNetNodes(NetNode* netNodes, Bucket* nodeDataBuckets) {
    NetNode& nn = netNodes[blockIdx.x];
    Bucket b = nodeDataBuckets[blockIdx.x];
    
    if (b.size) {
        for (size_t i = 0; i < b.size; ++i) {
            NetNode* p = netNodes + b.data[i];
            nn.inputs[i] = p;
            nn.outputs[i] = p;
        }
    }
}

__global__ void disposeOfBuckets(Bucket* buckets) {
    delete[] buckets[blockIdx.x].data;
}

__global__ void disposeOfBuckets(BucketTS* buckets) {
    delete[] buckets[blockIdx.x].data;
}

BrendanCUDA::Nets::Net BrendanCUDA::Nets::MakeNet_3D(size_t NodeCount, float ConnectionRange, BrendanCUDA::Random::AnyRNG<uint64_t> RNG, thrust::device_vector<float_3>** NodePoints) {
    thrust::device_vector<float_3>* dv = new thrust::device_vector<float_3>(NodeCount);

    fillRandomly<<<dv->size(), 1>>>(dv->data().get(), RNG());

    constexpr size_t bucketCountPerD = 10;

    BucketTS* bucketData;
    ThrowIfBad(cudaMalloc(&bucketData, bucketCountPerD * bucketCountPerD * bucketCountPerD * sizeof(BucketTS)));

    initBuckets<<<bucketCountPerD * bucketCountPerD * bucketCountPerD, 1>>>(bucketData);

    fillBuckets1<<<NodeCount, 1>>>(bucketData, bucketCountPerD, dv->data().get(), dv->size());

    device_vector<Bucket> nodesData(NodeCount);
    
    initBuckets<<<NodeCount, 1>>>(nodesData.data().get());

    fillBuckets2<<<NodeCount, 1>>>(nodesData.data().get(), bucketData, bucketCountPerD, dv->data().get(), ConnectionRange);

    device_vector<NetNode>& ndv = *new device_vector<NetNode>(NodeCount);

    for (size_t i = 0; i < NodeCount; ++i) {
        size_t c = ((Bucket)nodesData[i]).size;
        NetNode** ipts;
        NetNode** opts;
        ThrowIfBad(cudaMalloc(&ipts, c * sizeof(NetNode*)));
        ThrowIfBad(cudaMalloc(&opts, c * sizeof(NetNode*)));
        NetNode nn;
        nn.data = 0;
        nn.inputCount = c;
        nn.outputCount = c;
        nn.inputs = ipts;
        nn.outputs = opts;
        ndv[i] = nn;
    }

    fillNetNodes<<<NodeCount, 1>>>(ndv.data().get(), nodesData.data().get());

    disposeOfBuckets<<<bucketCountPerD * bucketCountPerD * bucketCountPerD, 1>>>(bucketData);
    disposeOfBuckets<<<NodeCount, 1>>>(nodesData.data().get());

    cudaFree(bucketData);

    if (NodePoints) {
        *NodePoints = dv;
    }
    else {
        delete dv;
    }

    return Net(ndv);
}