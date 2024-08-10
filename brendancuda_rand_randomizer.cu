#include "brendancuda_rand_randomizer.h"
#include "brendancuda_arrays.h"
#include <curand_kernel.h>
#include <device_launch_parameters.h>

__global__ void randomizeArrayKernel(BrendanCUDA::Span<float> Array, float Scalar, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    float* l = Array.ptr + idx * Count;
    float* u = l + Count;
    for (; l < u; ++l)
        *l += Scalar * (curand_uniform(&state) - 0.5f);
}
__global__ void randomizeArrayKernel(BrendanCUDA::Span<double> Array, double Scalar, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    double* l = Array.ptr + idx * Count;
    double* u = l + Count;
    for (; l < u; ++l)
        *l += Scalar * (curand_uniform(&state) - 0.5);
}
__global__ void randomizeArrayKernel(BrendanCUDA::Span<float> Array, float Scalar, float Lower, float Upper, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    float* l = Array.ptr + idx * Count;
    float* u = l + Count;
    for (; l < u; ++l)
        *l += std::clamp(*l + curand_uniform(&state) * Scalar, Lower, Upper);
}
__global__ void randomizeArrayKernel(BrendanCUDA::Span<double> Array, double Scalar, double Lower, double Upper, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    double* l = Array.ptr + idx * Count;
    double* u = l + Count;
    for (; l < u; ++l)
        *l += std::clamp(*l + curand_uniform(&state) * Scalar, Lower, Upper);
}
__global__ void randomizeArrayWFlipsKernel(BrendanCUDA::Span<uint32_t> Array, uint32_t FlipProb, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = l + Count;
    for (; l < u; ++l)
        *l = BrendanCUDA::Random::RandomizeWFlips(*l, FlipProb, state);
}
__global__ void randomizeArrayWTargetsKernel(BrendanCUDA::Span<uint32_t> Array, uint32_t EachFlipProb, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = l + Count;
    for (; l < u; ++l)
        *l = BrendanCUDA::Random::RandomizeWTargets(*l, EachFlipProb, state);
}
__global__ void randomizeArrayWMutationsKernel(BrendanCUDA::Span<uint32_t> Array, uint32_t MutationProb, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = l + Count;
    for (; l < u; ++l)
        *l = BrendanCUDA::Random::RandomizeWMutations(*l, MutationProb, state);
}
__global__ void randomizeArrayKernel(BrendanCUDA::Span<uint32_t> Array, uint32_t Flips_FlipProb, uint32_t Targets_EachFlipProb, uint32_t Mutations_MutationProb, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = l + Count;
    for (; l < u; ++l)
        if (curand(&state) < Mutations_MutationProb)
            *l = BrendanCUDA::Random::RandomizeWFlips(BrendanCUDA::Random::RandomizeWTargets(*l, Targets_EachFlipProb, state), Flips_FlipProb, state);
        else
            *l = curand(&state);
}

__global__ void initArrayKernel(BrendanCUDA::Span<float> Array, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    float* l = Array.ptr + idx * Count;
    float* u = l + Count;
    for (; l < u; ++l)
        *l = curand_uniform(&state);
}
__global__ void initArrayKernel(BrendanCUDA::Span<double> Array, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    double* l = Array.ptr + idx * Count;
    double* u = l + Count;
    for (; l < u; ++l)
        *l = curand_uniform(&state);
}
__global__ void initArrayKernel(BrendanCUDA::Span<float> Array, float Lower, float Range, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    float* l = Array.ptr + idx * Count;
    float* u = l + Count;
    for (; l < u; ++l)
        *l = curand_uniform(&state) * Range + Lower;
}
__global__ void initArrayKernel(BrendanCUDA::Span<double> Array, double Lower, double Range, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    double* l = Array.ptr + idx * Count;
    double* u = l + Count;
    for (; l < u; ++l)
        *l = curand_uniform(&state) * Range + Lower;
}
__global__ void initArrayKernel(BrendanCUDA::Span<uint32_t> Array, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    for (; l < u; ++l)
        *l = curand(&state);
}
__global__ void initArrayKernel(BrendanCUDA::Span<uint32_t> Array, uint32_t ProbOf1, uint64_t Seed, uint64_t Count) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= Array.size)
        return;
    curandState state;
    curand_init(Seed, idx, 0, &state);
    uint32_t* l = Array.ptr + idx * Count;
    uint32_t* u = std::min(l + Count, Array.ptr + Array.size);
    for (; l < u; ++l)
        *l = BrendanCUDA::Random::Get32Bits(ProbOf1, state);
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

void BrendanCUDA::details::RandomizeArray_CallKernel(Span<float> Array, float Scalar, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar, Seed, elementsPerThread);
}
void BrendanCUDA::details::RandomizeArray_CallKernel(Span<double> Array, double Scalar, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar, Seed, elementsPerThread);
}
void BrendanCUDA::details::RandomizeArray_CallKernel(Span<float> Array, float Scalar, float LowerBound, float UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar, LowerBound, UpperBound, Seed, elementsPerThread);
}
void BrendanCUDA::details::RandomizeArray_CallKernel(Span<double> Array, double Scalar, double LowerBound, double UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Scalar, LowerBound, UpperBound, Seed, elementsPerThread);
}
void BrendanCUDA::details::RandomizeArrayWFlips_CallKernel(Span<uint32_t> Array, uint32_t FlipProbability, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayWFlipsKernel<<<blockCount, threadsPerBlock>>>(Array, FlipProbability, Seed, elementsPerThread);
}
void BrendanCUDA::details::RandomizeArrayWTargets_CallKernel(Span<uint32_t> Array, uint32_t EachFlipProbability, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayWTargetsKernel<<<blockCount, threadsPerBlock>>>(Array, EachFlipProbability, Seed, elementsPerThread);
}
void BrendanCUDA::details::RandomizeArrayWMutations_CallKernel(Span<uint32_t> Array, uint32_t MutationProbability, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayWMutationsKernel<<<blockCount, threadsPerBlock>>>(Array, MutationProbability, Seed, elementsPerThread);
}
void BrendanCUDA::details::RandomizeArray_CallKernel(Span<uint32_t> Array, uint32_t FlipProbability, uint32_t TargetFlipProbability, uint32_t MutationProbability, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    randomizeArrayKernel<<<blockCount, threadsPerBlock>>>(Array, FlipProbability, TargetFlipProbability, MutationProbability, Seed, elementsPerThread);
}
void BrendanCUDA::details::InitArray_CallKernel(Span<float> Array, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Seed, elementsPerThread);
}
void BrendanCUDA::details::InitArray_CallKernel(Span<double> Array, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Seed, elementsPerThread);
}
void BrendanCUDA::details::InitArray_CallKernel(Span<float> Array, float LowerBound, float UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, LowerBound, UpperBound - LowerBound, Seed, elementsPerThread);
}
void BrendanCUDA::details::InitArray_CallKernel(Span<double> Array, double LowerBound, double UpperBound, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, LowerBound, UpperBound - LowerBound, Seed, elementsPerThread);
}
void BrendanCUDA::details::InitArray_CallKernel(Span<uint32_t> Array, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, Seed, elementsPerThread);
}
void BrendanCUDA::details::InitArray_CallKernel(Span<uint32_t> Array, uint32_t ProbabilityOf1, uint64_t Seed) {
    uint32_t elementsPerThread;
    uint32_t threadsPerBlock;
    uint32_t blockCount;
    getKernelLaunchParams(Array.size, elementsPerThread, threadsPerBlock, blockCount);
    initArrayKernel<<<blockCount, threadsPerBlock>>>(Array, ProbabilityOf1, Seed, elementsPerThread);
}