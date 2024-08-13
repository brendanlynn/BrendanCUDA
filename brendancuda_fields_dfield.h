#pragma once

#include "brendancuda_errorhelp.h"
#include "brendancuda_fields_field.h"
#include "brendancuda_points.h"
#include <stdexcept>
#include <string>

namespace BrendanCUDA {
    namespace Fields {
        template <typename _T, size_t _DimensionCount>
        class DField final {
            static_assert(_DimensionCount, "_DimensionCount may not be zero.");
            using vector_t = FixedVector<uint32_t, _DimensionCount>;
        public:
            __host__ __device__ __forceinline DField(vector_t Dimensions);
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DField(_Ts... Dimensions)
                : DField(vector_t(Dimensions...)) { }
            __host__ __device__ __forceinline DField(vector_t Dimensions, _T* All);
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline DField(_Ts... Dimensions, _T* All)
                : DField(vector_t(Dimensions...), All) { }

            __host__ __device__ __forceinline uint32_t LengthX() const requires (_DimensionCount <= 4);
            __host__ __device__ __forceinline uint32_t LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4);
            __host__ __device__ __forceinline uint32_t LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4);
            __host__ __device__ __forceinline uint32_t LengthW() const requires (_DimensionCount == 4);

            template <size_t _Index>
                requires (_Index < _DimensionCount)
            __host__ __device__ __forceinline uint32_t Length() const;

            __host__ __device__ __forceinline vector_t Dimensions() const;

            __host__ __device__ __forceinline size_t ValueCount() const;
            __host__ __device__ __forceinline size_t SizeOnGPU() const;

            __host__ __device__ __forceinline void Dispose();

            __host__ __device__ __forceinline Field<_T, _DimensionCount> FFront() const;
            __host__ __device__ __forceinline Field<_T, _DimensionCount> FBack() const;
            __host__ __device__ __forceinline void Reverse();

            __host__ __device__ __forceinline void CopyAllIn(_T* All);
            __host__ __device__ __forceinline void CopyAllOut(_T* All) const;
            __host__ __device__ __forceinline void CopyValueIn(uint64_t Index, _T* Value);
            __host__ __device__ __forceinline void CopyValueIn(vector_t Coordinates, _T* Value);
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline void CopyValueIn(_Ts... Coordinates, _T* Value) {
                FBack().CopyValueIn(vector_t(Coordinates...), Value);
            }
            __host__ __device__ __forceinline void CopyValueOut(uint64_t Index, _T* Value) const;
            __host__ __device__ __forceinline void CopyValueOut(vector_t Coordinates, _T* Value) const;
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline void CopyValueOut(_Ts... Coordinates, _T* Value) const {
                FFront().CopyValueOut(vector_t(Coordinates...), Value);
            }

            template <bool _CopyToHost>
            __host__ __forceinline _T* GetAll() const;
#ifdef __CUDACC__
            __device__ __forceinline _T* GetAll() const;
#endif
            __host__ __device__ __forceinline void SetAll(_T* All);

            __host__ __device__ __forceinline _T GetValueAt(uint64_t Index) const;
            __host__ __device__ __forceinline _T GetValueAt(vector_t Coordinates) const;
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline _T GetValueAt(_Ts... Coordinates) const {
                return FFront().GetValueAt(vector_t(Coordinates...));
            }

            __host__ __device__ __forceinline void SetValueAt(uint64_t Index, _T Value);
            __host__ __device__ __forceinline void SetValueAt(vector_t Coordinates, _T Value);
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline void SetValueAt(_Ts... Coordinates, _T Value) {
                FBack().SetValueAt(vector_t(Coordinates...), Value);
            }

            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(vector_t Coordinates) const;
            template <std::convertible_to<uint32_t>... _Ts>
                requires (sizeof...(_Ts) == _DimensionCount)
            __host__ __device__ __forceinline uint64_t CoordinatesToIndex(_Ts... Coordinates) const {
                return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), vector_t(Coordinates...));
            }
            __host__ __device__ __forceinline vector_t IndexToCoordinates(uint64_t Index) const;

            __host__ __device__ __forceinline void FillWith(_T Value);

            __host__ __device__ __forceinline void CopyBlockIn(_T* Input, vector_t InputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates);
            __host__ __device__ __forceinline void CopyBlockOut(_T* Output, vector_t OutputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates);

            __forceinline size_t SerializedSize() const requires BSerializer::Serializable<_T>;
            __forceinline void Serialize(void*& Data) const requires BSerializer::Serializable<_T>;
            static __forceinline DField<_T, _DimensionCount> Deserialize(const void*& Data) requires BSerializer::Serializable<_T>;
            static __forceinline void Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T>;
        private:
            vector_t dimensions;

            _T* cudaArrayF;
            _T* cudaArrayB;
        };
    }
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField<_T, _DimensionCount>::DField(vector_t Dimensions) {
    for (size_t i = 0; i < _DimensionCount; ++i)
        if (!Dimensions[i]) {
            dimensions = vector_t();
            cudaArrayF = 0;
            cudaArrayB = 0;
            return;
        }
    dimensions = Dimensions;
    size_t dataSize = SizeOnGPU();
#ifdef __CUDA_ARCH__
    cudaArrayF = (_T*)malloc(dataSize);
    cudaArrayB = (_T*)malloc(dataSize);
#else
    ThrowIfBad(cudaMalloc(&cudaArrayF, dataSize));
    ThrowIfBad(cudaMalloc(&cudaArrayB, dataSize));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline BrendanCUDA::Fields::DField<_T, _DimensionCount>::DField(vector_t Dimensions, _T* All)
    : DField(Dimensions) {
    FFront().CopyAllIn(All);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::Dispose() {
#ifdef __CUDA_ARCH__
    free(cudaArrayF);
    free(cudaArrayB);
#else
    ThrowIfBad(cudaFree(cudaArrayF));
    ThrowIfBad(cudaFree(cudaArrayB));
#endif
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field<_T, _DimensionCount> BrendanCUDA::Fields::DField<_T, _DimensionCount>::FFront() const {
    return *(Field<_T, _DimensionCount>*)this;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline BrendanCUDA::Fields::Field<_T, _DimensionCount> BrendanCUDA::Fields::DField<_T, _DimensionCount>::FBack() const {
    uint8_t r[sizeof(Field<_T, _DimensionCount>)];
    *(uint64_t*)r = *(uint64_t*)this;
    ((uint32_t*)r)[2] = ((uint32_t*)this)[2];
    ((void**)r)[2] = ((void**)this)[3];
    return *(Field<_T, _DimensionCount>*)&r;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::Reverse() {
    _T* i = cudaArrayF;
    cudaArrayF = cudaArrayB;
    cudaArrayB = i;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::LengthX() const requires (_DimensionCount <= 4) {
    return dimensions.x;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::LengthY() const requires (_DimensionCount >= 2 && _DimensionCount <= 4) {
    return dimensions.y;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::LengthZ() const requires (_DimensionCount >= 3 && _DimensionCount <= 4) {
    return dimensions.z;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::LengthW() const requires (_DimensionCount == 4) {
    return dimensions.w;
}
template <typename _T, size_t _DimensionCount>
template <size_t _Index>
    requires (_Index < _DimensionCount)
__host__ __device__ __forceinline uint32_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::Length() const {
    return dimensions[_Index];
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::Fields::DField<_T, _DimensionCount>::Dimensions() const -> vector_t {
    return dimensions;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::ValueCount() const {
    size_t s = 1;
    for (size_t i = 0; i < _DimensionCount; ++i)
        s *= dimensions[i];
    return s;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline size_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::SizeOnGPU() const {
    return (sizeof(_T) * ValueCount()) << 1;
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline uint64_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::CoordinatesToIndex(vector_t Coordinates) const {
    return BrendanCUDA::CoordinatesToIndex<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Coordinates);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline auto BrendanCUDA::Fields::DField<_T, _DimensionCount>::IndexToCoordinates(uint64_t Index) const -> vector_t {
    return BrendanCUDA::IndexToCoordinates<uint64_t, uint32_t, _DimensionCount, true>(Dimensions(), Index);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::FillWith(_T Value) {
    FBack().FillWith(Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyAllIn(_T* All) {
    FBack().CopyAllIn(All);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyAllOut(_T* All) const {
    FFront().CopyAllOut(All);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyValueIn(uint64_t Index, _T* Value) {
    FBack().CopyValueIn(Index, Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyValueIn(vector_t Coordinates, _T* Value) {
    FBack().CopyValueIn(Coordinates, Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyValueOut(uint64_t Index, _T* Value) const {
    FFront().CopyValueOut(Index, Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyValueOut(vector_t Coordinates, _T* Value) const {
    FFront().CopyValueOut(Coordinates, Value);
}
template <typename _T, size_t _DimensionCount>
template <bool _CopyToHost>
__host__ __forceinline _T* BrendanCUDA::Fields::DField<_T, _DimensionCount>::GetAll() const {
    return FFront().GetAll<_CopyToHost>();
}
#ifdef __CUDACC__
template <typename _T, size_t _DimensionCount>
__device__ __forceinline _T* BrendanCUDA::Fields::DField<_T, _DimensionCount>::GetAll() const {
    return FFront().GetAll();
}
#endif
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::SetAll(_T* All) {
    FBack().SetAll(All);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField<_T, _DimensionCount>::GetValueAt(uint64_t Index) const {
    return FFront().GetValueAt(Index);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline _T BrendanCUDA::Fields::DField<_T, _DimensionCount>::GetValueAt(vector_t Coordinates) const {
    return FFront().GetValueAt(Coordinates);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::SetValueAt(uint64_t Index, _T Value) {
    FBack().SetValueAt(Index, Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::SetValueAt(vector_t Coordinates, _T Value) {
    FBack().SetValueAt(Coordinates, Value);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyBlockIn(_T* Input, vector_t InputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates) {
    FBack().CopyBlockIn(Input, InputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T, size_t _DimensionCount>
__host__ __device__ __forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::CopyBlockOut(_T* Output, vector_t OutputDimensions, vector_t RangeDimensions, vector_t RangeInInputsCoordinates, vector_t RangeInOutputsCoordinates) {
    FFront().CopyBlockOut(Output, OutputDimensions, RangeDimensions, RangeInInputsCoordinates, RangeInOutputsCoordinates);
}
template <typename _T, size_t _DimensionCount>
__forceinline size_t BrendanCUDA::Fields::DField<_T, _DimensionCount>::SerializedSize() const requires BSerializer::Serializable<_T> {
    return FFront().SerializedSize();
}
template <typename _T, size_t _DimensionCount>
__forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::Serialize(void*& Data) const requires BSerializer::Serializable<_T> {
    FFront().Serialize(Data);
}
template <typename _T, size_t _DimensionCount>
__forceinline auto BrendanCUDA::Fields::DField<_T, _DimensionCount>::Deserialize(const void*& Data) -> DField<_T, _DimensionCount> requires BSerializer::Serializable<_T> {
    vector_t dimensions = BSerializer::Deserialize<vector_t>(Data);
    DField<_T, _DimensionCount> dfield(dimensions);
    Field<_T, _DimensionCount> field = dfield.FFront();
    size_t l = field.ValueCount();
    for (size_t i = 0; i < l; ++i)
        field.SetValueAt(i, BSerializer::Deserialize<_T>(Data));
    return field;
}
template <typename _T, size_t _DimensionCount>
__forceinline void BrendanCUDA::Fields::DField<_T, _DimensionCount>::Deserialize(const void*& Data, void* Value) requires BSerializer::Serializable<_T> {
    new (Value) DField<_T, _DimensionCount>(Deserialize(Data));
}