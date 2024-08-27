#include "ai_mlp_mlp.h"

#pragma warning(disable : 4996)
__global__ void runActivationFunctionOnArrayKernel(float* Array, BrendanCUDA::AI::activationFunction_t<float> ActivationFunction) {
    float& p(Array[blockIdx.x]);
    p = ActivationFunction(p);
}
__global__ void runActivationFunctionOnArrayKernel(double* Array, BrendanCUDA::AI::activationFunction_t<double> ActivationFunction) {
    double& p(Array[blockIdx.x]);
    p = ActivationFunction(p);
}
template <typename _T>
__host__ __device__ void BrendanCUDA::details::RunActivationFunctionOnArray(Span<_T> Array, AI::activationFunction_t<_T> ActivationFunction) {
#ifdef __CUDA_ARCH__
    for (size_t i = 0; i < Array.size; ++i) {
        _T& p(Array[i]);
        p = ActivationFunction(p);
    }
#else
    runActivationFunctionOnArrayKernel<<<Array.size, 1>>>(Array.ptr, ActivationFunction);
#endif
}

template void BrendanCUDA::details::RunActivationFunctionOnArray<float>(Span<float>, AI::activationFunction_t<float>);
template void BrendanCUDA::details::RunActivationFunctionOnArray<double>(Span<double>, AI::activationFunction_t<double>);