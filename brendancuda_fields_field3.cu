#include "brendancuda_fields_field3.h"

template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field3<_T>::FillWith(_T Value) {
    details::fillWithKernel<_T><<<lengthX * lengthY * lengthZ, 1>>>(cudaArray, Value);
}