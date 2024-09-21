#include "arrays.h"
#include "rand_randomizer.h"
#include <curand_kernel.h>
#include <device_launch_parameters.h>

__global__ void randomizeArrayKernel(bcuda::Span<float> Array, float Scalar, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    float* l = Array.ptr + idx * Count;
    float* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l += Scalar * (curand_uniform(&state) - 0.5f);
    while (++l < u);
}
__global__ void randomizeArrayKernel(bcuda::Span<double> Array, double Scalar, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    double* l = Array.ptr + idx * Count;
    double* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l += Scalar * (curand_uniform(&state) - 0.5);
    while (++l < u);
}
__global__ void randomizeArrayKernel(bcuda::Span<float> Array, float Scalar, float Lower, float Upper, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    float* l = Array.ptr + idx * Count;
    float* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l += std::clamp(*l + curand_uniform(&state) * Scalar, Lower, Upper);
    while (++l < u);
}
__global__ void randomizeArrayKernel(bcuda::Span<double> Array, double Scalar, double Lower, double Upper, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    double* l = Array.ptr + idx * Count;
    double* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l += std::clamp(*l + curand_uniform(&state) * Scalar, Lower, Upper);
    while (++l < u);
}
__global__ void randomizeArrayWFlipsKernel(bcuda::Span<uint32_t> Array, uint32_t FlipProb, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = bcuda::rand::RandomizeWFlips(*l, FlipProb, state);
    while (++l < u);
}
__global__ void randomizeArrayWTargetsKernel(bcuda::Span<uint32_t> Array, uint32_t EachFlipProb, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = bcuda::rand::RandomizeWTargets(*l, EachFlipProb, state);
    while (++l < u);
}
__global__ void randomizeArrayWMutationsKernel(bcuda::Span<uint32_t> Array, uint32_t MutationProb, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = bcuda::rand::RandomizeWMutations(*l, MutationProb, state);
    while (++l < u);
}
__global__ void randomizeArrayWMutationsKernel(bcuda::Span<uint32_t> Array, uint32_t MutationProb, uint32_t ProbabilityOf1, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = bcuda::rand::RandomizeWMutations(*l, MutationProb, ProbabilityOf1, state);
    while (++l < u);
}

__global__ void initArrayKernel(bcuda::Span<float> Array, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    float* l = Array.ptr + idx * Count;
    float* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = curand_uniform(&state);
    while (++l < u);
}
__global__ void initArrayKernel(bcuda::Span<double> Array, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    double* l = Array.ptr + idx * Count;
    double* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = curand_uniform(&state);
    while (++l < u);
}
__global__ void initArrayKernel(bcuda::Span<float> Array, float Lower, float Range, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    float* l = Array.ptr + idx * Count;
    float* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = curand_uniform(&state) * Range + Lower;
    while (++l < u);
}
__global__ void initArrayKernel(bcuda::Span<double> Array, double Lower, double Range, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    double* l = Array.ptr + idx * Count;
    double* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = curand_uniform(&state) * Range + Lower;
    while (++l < u);
}
__global__ void initArrayKernel(bcuda::Span<uint32_t> Array, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = curand(&state);
    while (++l < u);
}
__global__ void initArrayKernel(bcuda::Span<uint32_t> Array, uint32_t ProbOf1, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    if (l >= u) return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    do *l = bcuda::rand::Get32Bits(ProbOf1, state);
    while (++l < u);
}
__global__ void clearArrayKernel(bcuda::Span<float> Array, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    float* l = Array.ptr + idx * Count;
    float* u = std::min(l + Count, Array.ptr + Array.size);
    for (; l < u; ++l)
        *l = 0.f;
}
__global__ void clearArrayKernel(bcuda::Span<double> Array, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    double* l = Array.ptr + idx * Count;
    double* u = std::min(l + Count, Array.ptr + Array.size);
    for (; l < u; ++l)
        *l = 0.;
}
__global__ void clearArrayKernel(bcuda::Span<uint64_t> Array, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t* l = Array.ptr + idx * Count;
    uint64_t* u = std::min(l + Count, Array.ptr + Array.size);
    for (; l < u; ++l)
        *l = 0;
}

void getKernelLaunchParams(uint64_t ElementCount, uint32_t& ElementsPerThread, uint32_t& ThreadsPerBlock, uint32_t& BlockCount) {
    int deviceIdx;
    cudaGetDevice(&deviceIdx);
    cudaDeviceProp dprops;
    cudaGetDeviceProperties(&dprops, deviceIdx);
    
    if (ElementCount <= dprops.maxThreadsPerBlock) {
        ElementsPerThread = 1;
        ThreadsPerBlock = (uint32_t)ElementCount;
        BlockCount = 1;
    }
    else {
        uint64_t maxConcurrency = dprops.maxThreadsPerMultiProcessor * (uint64_t)dprops.multiProcessorCount;
        uint64_t secondPerThreadBound = maxConcurrency + (maxConcurrency >> 1) + (maxConcurrency >> 2);
        if (ElementCount <= secondPerThreadBound) {
            ElementsPerThread = 1;
            ThreadsPerBlock = dprops.maxThreadsPerBlock;
            BlockCount = (uint32_t)((ElementCount + dprops.maxThreadsPerBlock - 1) / dprops.maxThreadsPerBlock);
        }
        else {
            ElementsPerThread = (uint32_t)((ElementCount + maxConcurrency - 1) / maxConcurrency);
            ThreadsPerBlock = dprops.maxThreadsPerBlock;
            BlockCount = dprops.maxBlocksPerMultiProcessor * dprops.multiProcessorCount;
        }
    }
}

void bcuda::details::RandomizeArray_CallKernel(Span<float> Array, float Scalar, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar * 2.f, Seed, elementsPerThread);
}
void bcuda::details::RandomizeArray_CallKernel(Span<double> Array, double Scalar, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar * 2., Seed, elementsPerThread);
}
void bcuda::details::RandomizeArray_CallKernel(Span<float> Array, float Scalar, float LowerBound, float UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar * 2.f, LowerBound, UpperBound, Seed, elementsPerThread);
}
void bcuda::details::RandomizeArray_CallKernel(Span<double> Array, double Scalar, double LowerBound, double UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar * 2., LowerBound, UpperBound, Seed, elementsPerThread);
}
void bcuda::details::RandomizeArrayWFlips_CallKernel(Span<uint32_t> Array, uint32_t FlipProbability, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayWFlipsKernel<<<blockCount, threadsPerBlock>>>(Array, FlipProbability, Seed, elementsPerThread);
}
void bcuda::details::RandomizeArrayWTargets_CallKernel(Span<uint32_t> Array, uint32_t EachFlipProbability, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayWTargetsKernel<<<blockCount, threadsPerBlock>>>(Array, EachFlipProbability, Seed, elementsPerThread);
}
void bcuda::details::RandomizeArrayWMutations_CallKernel(Span<uint32_t> Array, uint32_t MutationProbability, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayWMutationsKernel<<<blockCount, threadsPerBlock>>>(Array, MutationProbability, Seed, elementsPerThread);
}
void bcuda::details::RandomizeArrayWMutations_CallKernel(Span<uint32_t> Array, uint32_t MutationProbability, uint32_t ProbabilityOf1, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayWMutationsKernel<<<blockCount, threadsPerBlock>>>(Array, MutationProbability, ProbabilityOf1, Seed, elementsPerThread);
}
void bcuda::details::InitArray_CallKernel(Span<float> Array, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Seed, elementsPerThread);
}
void bcuda::details::InitArray_CallKernel(Span<double> Array, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Seed, elementsPerThread);
}
void bcuda::details::InitArray_CallKernel(Span<float> Array, float LowerBound, float UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, LowerBound, UpperBound - LowerBound, Seed, elementsPerThread);
}
void bcuda::details::InitArray_CallKernel(Span<double> Array, double LowerBound, double UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, LowerBound, UpperBound - LowerBound, Seed, elementsPerThread);
}
void bcuda::details::InitArray_CallKernel(Span<uint32_t> Array, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Seed, elementsPerThread);
}
void bcuda::details::InitArray_CallKernel(Span<uint32_t> Array, uint32_t ProbabilityOf1, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, ProbabilityOf1, Seed, elementsPerThread);
}
void bcuda::details::ClearArray_CallKernel(Span<float> Array) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    clearArrayKernel<<<blockCount, threadsPerBlock>>>(Array, elementsPerThread);
}
void bcuda::details::ClearArray_CallKernel(Span<double> Array) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    clearArrayKernel<<<blockCount, threadsPerBlock>>>(Array, elementsPerThread);
}
void bcuda::details::ClearArray_CallKernel(Span<uint64_t> Array) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    clearArrayKernel<<<blockCount, threadsPerBlock>>>(Array, elementsPerThread);
}