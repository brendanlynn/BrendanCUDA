#include "brendancuda_fields_field2.h"

template <typename _T>
__host__ __device__ __forceinline void BrendanCUDA::Fields::Field2<_T>::FillWith(_T Value) {
    details::fillWithKernel<_T><<<lengthX * lengthY, 1>>>(cudaArray, Value);
}