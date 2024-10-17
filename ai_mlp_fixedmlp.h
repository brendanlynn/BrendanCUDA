#pragma once

#include "ai.h"
#include "BSerializer/Serializer.h"
#include "curandkernelgens.h"
#include "errorhelp.h"
#include "mathfuncs.h"
#include "rand_anyrng.h"
#include "rand_randomizer.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>
#include <type_traits>

namespace bcuda {
    namespace ai {
        namespace mlp {
            template <std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _OutputCount>
                requires ai::IsActivationFunction<_ActivationFunction, _T>
            struct FixedMLPL;
        }
    }
    namespace details {
        template <size_t _Index, std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _ContinuedOutputCounts>
        struct MLPLayerType;
        template <size_t _Index, std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
        struct MLPLayerType<_Index, _T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> {
            using type = typename MLPLayerType<_Index - 1, _T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::type;
        };
        template <std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
        struct MLPLayerType<0, _T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> {
            using type = ai::mlp::FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count>;
        };
        template <size_t _Index, std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count>
        struct MLPLayerType<_Index, _T, _ActivationFunction, _InputCount, _Output1Count> {
            static_assert(!_Index, "_Index is out of bounds.");
            using type = ai::mlp::FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count>;
        };

        template <size_t _Index, std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _ContinuedOutputCounts>
        using mlpLayerType_t = MLPLayerType<_Index, _T, _ActivationFunction, _InputCount, _Output1Count, _ContinuedOutputCounts...>;

        template <uintmax_t _Idx, size_t... _Ints>
        struct getIntsByIndex;
        template <size_t _Int1, size_t... _ContinuedInts>
        struct getIntsByIndex<0, _Int1, _ContinuedInts...> {
            static constexpr size_t value = _Int1;
        };
        template <uintmax_t _Idx, size_t _Int1, size_t... _ContinuedInts>
        struct getIntsByIndex<_Idx, _Int1, _ContinuedInts...> {
            static constexpr size_t value = getIntsByIndex<_Idx - 1, _ContinuedInts...>::value;
        };
    }
    namespace ai {
        namespace mlp {
            template <std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _OutputCount>
                requires ai::IsActivationFunction<_ActivationFunction, _T>
            struct FixedMLPL {
                static_assert(_InputCount, "_InputCount must be greater than 0.");
                static_assert(_OutputCount, "_OutputCount must be greater than 0.");
            private:
                using this_t = FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>;
            public:
                using element_t = _T;
                static constexpr auto activationFunction = _ActivationFunction;
                static constexpr size_t inputCount = _InputCount;
                static constexpr size_t outputCount = _OutputCount;

                _T weights[_InputCount][_OutputCount];
                _T bias[_OutputCount];

                __host__ __device__ inline void FillWith0() {
                    for (size_t i = 0; i < _OutputCount; ++i) {
                        for (size_t j = 0; j < _InputCount; ++j)
                            weights[i][j] = 0.;
                        bias[i] = 0.;
                    }
                }
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void FillWithRandom(_TRNG& RNG) {
                    std::uniform_real_distribution<_T> dis(-1., 1.);

                    for (size_t i = 0; i < _OutputCount; ++i) {
                        for (size_t j = 0; j < _InputCount; ++j)
                            weights[i][j] = dis(RNG);
                        bias[i] = dis(RNG);
                    }
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void FillWithRandom(_TRNG& RNG) {
                    if constexpr (std::same_as<_T, float>)
                        for (size_t i = 0; i < _OutputCount; ++i) {
                            for (size_t j = 0; j < _InputCount; ++j)
                                weights[i][j] = curand_uniform(&RNG) * 2.f - 1.f;
                            bias[i] = curand_uniform(&RNG) * 2.f - 1.f;
                        }
                    else
                        for (size_t i = 0; i < _OutputCount; ++i) {
                            for (size_t j = 0; j < _InputCount; ++j)
                                weights[i][j] = curand_uniform_double(&RNG) * 2. - 1.;
                            bias[i] = curand_uniform_double(&RNG) * 2. - 1.;
                        }
                 }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void ChangeWithRandom(_T Scalar, _TRNG& RNG) {
                    std::uniform_real_distribution<_T> dis(-Scalar, Scalar);

                    for (size_t i = 0; i < _OutputCount; ++i) {
                        for (size_t j = 0; j < _InputCount; ++j)
                            weights[i][j] += dis(RNG);
                        bias[i] += dis(RNG);
                    }
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void ChangeWithRandom(_T Scalar, _TRNG& RNG) {
                    if constexpr (std::same_as<_T, float>)
                        for (size_t i = 0; i < _OutputCount; ++i) {
                            for (size_t j = 0; j < _InputCount; ++j)
                                weights[i][j] = curand_uniform(&RNG) * 2.f - 1.f;
                            bias[i] = curand_uniform(&RNG) * 2.f - 1.f;
                        }
                    else
                        for (size_t i = 0; i < _OutputCount; ++i) {
                            for (size_t j = 0; j < _InputCount; ++j)
                                weights[i][j] = curand_uniform_double(&RNG) * 2. - 1.;
                            bias[i] = curand_uniform_double(&RNG) * 2. - 1.;
                        }
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    std::uniform_real_distribution<_T> dis(-Scalar, Scalar);

                    for (size_t i = 0; i < _OutputCount; ++i) {
                        for (size_t j = 0; j < _InputCount; ++j) {
                            _T& v = weights[i][j];
                            v = math::clamp(v + dis(RNG), LowerBound, UpperBound);
                        }
                        _T& v = bias[i];
                        v = math::clamp(v + dis(RNG), LowerBound, UpperBound);
                    }
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    if constexpr (std::same_as<_T, float>)
                        for (size_t i = 0; i < _OutputCount; ++i) {
                            for (size_t j = 0; j < _InputCount; ++j) {
                                _T& v = weights[i][j];
                                v = math::clamp(v + curand_uniform(&RNG), LowerBound, UpperBound);
                            }
                            _T& v = bias[i];
                            v = math::clamp(v + curand_uniform(&RNG), LowerBound, UpperBound);
                        }
                    else
                        for (size_t i = 0; i < _OutputCount; ++i) {
                            for (size_t j = 0; j < _InputCount; ++j) {
                                _T& v = weights[i][j];
                                v = math::clamp(v + curand_uniform_double(&RNG), LowerBound, UpperBound);
                            }
                            _T& v = bias[i];
                            v = math::clamp(v + curand_uniform_double(&RNG), LowerBound, UpperBound);
                        }
                }
#endif
                __host__ __device__ void Run(const _T* Input, _T* Output) const {
                    if (Input == Output) {
                        _T* secondOutput = new _T[_OutputCount];
                        for (size_t j = 0; j < _OutputCount; ++j) {
                            _T v = bias[j];
                            for (size_t i = 0; i < _InputCount; ++i)
                                v += weights[i][j] * Input[i];
                            secondOutput[j] = _ActivationFunction(v);
                        }
                        memcpy(Output, secondOutput, sizeof(_T) * _OutputCount);
                    }
                    else {
                        for (size_t j = 0; j < _OutputCount; ++j) {
                            _T v = bias[j];
                            for (size_t i = 0; i < _InputCount; ++i)
                                v += weights[i][j] * Input[i];
                            Output[j] = _ActivationFunction(v);
                        }
                    }
                }
                __host__ __device__ consteval std::array<_T, _OutputCount> Run(const _T(&Input)[_InputCount]) {
                    std::array<_T, _OutputCount> output;
                    for (size_t j = 0; j < _OutputCount; ++j) {
                        _T v = bias[j];
                        for (size_t i = 0; i < _InputCount; ++i)
                            v += weights[i][j] * Input[i];
                        output[j] = _ActivationFunction(v);
                    }
                    return output;
                }

                inline size_t SerializedSize() const {
                    return sizeof(this_t);
                }
                inline void Serialize(void*& Data) const {
                    BSerializer::SerializeArray(Data, weights, _InputCount * _OutputCount);
                    BSerializer::SerializeArray(Data, bias, _OutputCount);
                }
                inline static this_t Deserialize(const void*& Data) {
                    uint8_t bytes[sizeof(this_t)];
                    Deserialize(Data, &bytes);
                    return *(this_t*)&bytes;
                }
                inline static void Deserialize(const void*& Data, void* ObjMem) {
                    this_t& obj = &(this_t*)ObjMem;
                    BSerializer::DeserializeArray(Data, obj.weights, _InputCount * _OutputCount);
                    BSerializer::DeserializeArray(Data, obj.bias, _OutputCount);
                }
            };
            template <std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
                requires ai::IsActivationFunction<_ActivationFunction, _T>
            struct FixedMLP;
            template <std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t _Output2Count, size_t... _ContinuedOutputCounts>
                requires ai::IsActivationFunction<_ActivationFunction, _T>
            struct FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...> {
            private:
                using this_t = FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>;
            public:
                using element_t = _T;
                static constexpr auto activationFunction = _ActivationFunction;
                template <size_t _Index>
                static constexpr size_t widthAt = details::getIntsByIndex<_Index, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::value;
                template <size_t _Index>
                using layerType_t = details::mlpLayerType_t<_Index, _T, _ActivationFunction, _InputCount, _Output1Count, _Output2Count, _ContinuedOutputCounts...>;

                FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count> layer;
                FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...> nextLayers;

                __host__ __device__ inline void FillWith0() {
                    layer.FillWith0();
                    nextLayers.FillWith0();
                }
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                    nextLayers.FillWithRandom(RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                    nextLayers.FillWithRandom(RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void ChangeWithRandom(_T Scalar, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, RNG);
                    nextLayers.ChangeWithRandom(Scalar, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void ChangeWithRandom(_T Scalar, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, RNG);
                    nextLayers.ChangeWithRandom(Scalar, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
                    nextLayers.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
                    nextLayers.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
                }
#endif

                template <size_t _Index>
                __host__ __device__ inline layerType_t<_Index>& Layer() {
                    if constexpr (_Index) {
                        return nextLayers.Layer<_Index - 1>();
                    }
                    else {
                        return layer;
                    }
                }

                static constexpr size_t InputCount() {
                    return _InputCount;
                }
                static constexpr size_t OutputCount() {
                    return bcuda::ai::mlp::FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::OutputCount();
                }
                static constexpr size_t Intermediate0Count() {
                    return std::max(_Output1Count, bcuda::ai::mlp::FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Intermediate1Count());
                }
                static constexpr size_t Intermediate1Count() {
                    return bcuda::ai::mlp::FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::Intermediate0Count();
                }
                static constexpr size_t MaxLayerOutputCount() {
                    constexpr size_t maxNextLayers = FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>::MaxLayerOutputCount();
                    return _Output1Count > maxNextLayers ? _Output1Count : maxNextLayers;
                }
                static constexpr size_t LayerCount() {
                    return sizeof...(_ContinuedOutputCounts) + 2;
                }

                __host__ __device__ inline void Run(const _T* Input, _T* Output) const {
                    Run(Input, 0, 0, Output);
                }
                __host__ __device__ inline void Run(const _T* Input, _T* Intermediate1, _T* Intermediate2, _T* Output) const {
                    _T* i1 = Intermediate1 ? Intermediate1 : new _T[Intermediate0Count()];
                    _T* i2 = Intermediate2 ? Intermediate2 : new _T[Intermediate1Count()];

                    layer.Run(Input, i1);
                    nextLayers.Run(i1, i2, i1, Output);

                    if (!Intermediate1) delete[] i1;
                    if (!Intermediate2) delete[] i2;
                }
                __host__ __device__ consteval std::array<_T, OutputCount()> Run(const _T(&Input)[_InputCount]) {
                    auto firstOutput = layer.Run(Input);
                    return nextLayers.Run(firstOutput);
                }

                inline size_t SerializedSize() const {
                    return sizeof(this_t);
                }
                inline void Serialize(void*& Data) const {
                    layer.Serialize(Data);
                    nextLayers.Serialize(Data);
                }
                inline static this_t Deserialize(const void*& Data) {
                    uint8_t bytes[sizeof(this_t)];
                    Deserialize(Data, &bytes);
                    return *(this_t*)&bytes;
                }
                inline static void Deserialize(const void*& Data, void* ObjMem) {
                    this_t& obj = &(this_t*)ObjMem;
                    BSerializer::Deserialize<layerType_t<0>>(Data, &obj.layer);
                    BSerializer::Deserialize<FixedMLP<_T, _ActivationFunction, _Output1Count, _Output2Count, _ContinuedOutputCounts...>>(Data, &obj.nextLayers);
                }
            };
            template <std::floating_point _T, auto _ActivationFunction, size_t _InputCount, size_t _Output1Count>
                requires ai::IsActivationFunction<_ActivationFunction, _T>
            struct FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count> {
            private:
                using this_t = FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count>;
            public:
                using element_t = _T;
                static constexpr auto activationFunction = _ActivationFunction;
                template <size_t _Index>
                static constexpr size_t widthAt = details::getIntsByIndex<_Index, _InputCount, _Output1Count>::value;
                template <size_t _Index>
                using layerType_t = details::mlpLayerType_t<_Index, _T, _ActivationFunction, _InputCount, _Output1Count>;

                FixedMLPL<_T, _ActivationFunction, _InputCount, _Output1Count> layer;

                __host__ __device__ inline void FillWith0() {
                    layer.FillWith0();
                }
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void FillWithRandom(_TRNG& RNG) {
                    layer.FillWithRandom(RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void ChangeWithRandom(_T Scalar, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void ChangeWithRandom(_T Scalar, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, RNG);
                }
#endif
                template <std::uniform_random_bit_generator _TRNG>
                __host__ inline void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
                }
#ifdef __CUDACC__
                template <KernelCurandState _TRNG>
                __device__ inline void ChangeWithRandom(_T Scalar, _T LowerBound, _T UpperBound, _TRNG& RNG) {
                    layer.ChangeWithRandom(Scalar, LowerBound, UpperBound, RNG);
                }
#endif
                template <size_t _Index>
                __host__ __device__ inline layerType_t<_Index>& Layer() {
                    static_assert(!_Index, "_Index is out of bounds (too large).");

                    return layer;
                }

                static constexpr size_t InputCount() {
                    return _InputCount;
                }
                static constexpr size_t OutputCount() {
                    return _Output1Count;
                }
                static constexpr size_t Intermediate0Count() {
                    return 0;
                }
                static constexpr size_t Intermediate1Count() {
                    return 0;
                }
                static constexpr size_t MaxLayerOutputCount() {
                    return _Output1Count;
                }
                static constexpr size_t LayerCount() {
                    return 1;
                }

                __host__ __device__ inline void Run(const _T* Input, _T* Output) const {
                    layer.Run(Input, Output);
                }
                __host__ __device__ inline void Run(const _T* Input, _T* Intermediate1, _T* Intermediate2, _T* Output) const {
                    layer.Run(Input, Output);
                }
                __host__ __device__ consteval std::array<_T, _OutputCount> Run(const _T(&Input)[_InputCount]) {
                    return layer.Run(Input);
                }

                inline size_t SerializedSize() const {
                    return sizeof(this_t);
                }
                inline void Serialize(void*& Data) const {
                    layer.Serialize(Data);
                }
                inline static this_t Deserialize(const void*& Data) {
                    uint8_t bytes[sizeof(this_t)];
                    Deserialize(Data, &bytes);
                    return *(this_t*)&bytes;
                }
                inline static void Deserialize(const void*& Data, void* ObjMem) {
                    this_t& obj = &(this_t*)ObjMem;
                    BSerializer::Deserialize<layerType_t<0>>(Data, &obj.layer);
                }
            };
        }
    }
    namespace details {
        template <typename _T>
        struct isFixedMLPL : std::false_type { };
        template <std::floating_point _T, ai::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _OutputCount>
        struct isFixedMLPL<ai::mlp::FixedMLPL<_T, _ActivationFunction, _InputCount, _OutputCount>> : std::true_type { };
        template <typename _T>
        struct isFixedMLP : std::false_type { };
        template <std::floating_point _T, ai::activationFunction_t<_T> _ActivationFunction, size_t _InputCount, size_t _Output1Count, size_t... _LayerCounts>
        struct isFixedMLP<ai::mlp::FixedMLP<_T, _ActivationFunction, _InputCount, _Output1Count, _LayerCounts...>> : std::true_type { };
    }
    namespace ai {
        namespace mlp {
            template <typename _T>
            concept IsFixedMLPL = details::isFixedMLPL<_T>::value;
            template <typename _T>
            concept IsFixedMLP = details::isFixedMLP<_T>::value;
        }
    }
    namespace details {
        template <ai::mlp::IsFixedMLP _TFixedMLP>
        inline static void FixedMLP_Run(const _TFixedMLP* Mlp, const typename _TFixedMLP::element_t* Inputs, typename _TFixedMLP::element_t* Intermediate0, typename _TFixedMLP::element_t* Intermediate1, typename _TFixedMLP::element_t* Outputs) {
            using element_t = typename _TFixedMLP::element_t;

            if constexpr (_TFixedMLP::LayerCount() == 1) {
                FixedMLPL_Run<decltype(Mlp->layer), false, false>(&Mlp->layer, Inputs, Outputs);
            }
            else {
                ai::mlp::FixedMLPL_Run<decltype(Mlp->layer), false, false>(&Mlp->layer, Inputs, Intermediate0);
                FixedMLP_Run(&Mlp->nextLayers, Intermediate0, Intermediate1, Intermediate0, Outputs);
            }
        }
    }
    namespace ai {
        namespace mlp {
            template <typename _TFixedMLPL>
                requires IsFixedMLPL<std::remove_const_t<_TFixedMLPL>>
            __host__ __device__ inline static Span<std::conditional_t<std::is_const_v<_TFixedMLPL>, const typename _TFixedMLPL::element_t, typename _TFixedMLPL::element_t>> FixedMLPL_GetElementSpan(_TFixedMLPL* MLPL) {
                using element_t = std::conditional_t<std::is_const_v<_TFixedMLPL>, const typename _TFixedMLPL::element_t, typename _TFixedMLPL::element_t>;
                return Span<element_t>((element_t*)MLPL, sizeof(_TFixedMLPL) / sizeof(element_t));
            }
            template <typename _TFixedMLP>
                requires IsFixedMLP<std::remove_const_t<_TFixedMLP>>
            __host__ __device__ inline static Span<std::conditional_t<std::is_const_v<_TFixedMLP>, const typename _TFixedMLP::element_t, typename _TFixedMLP::element_t>> FixedMLP_GetElementSpan(_TFixedMLP* MLP) {
                using element_t = std::conditional_t<std::is_const_v<_TFixedMLP>, const typename _TFixedMLP::element_t, typename _TFixedMLP::element_t>;
                return Span<element_t>((element_t*)MLP, sizeof(_TFixedMLP) / sizeof(element_t));
            }

            template <IsFixedMLPL _TFixedMLPL, bool _InputOnHost, bool _OutputOnHost>
            inline static void FixedMLPL_Run(const _TFixedMLPL* MLPL, const typename _TFixedMLPL::element_t* Inputs, typename _TFixedMLPL::element_t* Outputs) {
                using element_t = typename _TFixedMLPL::element_t;
                if constexpr (_InputOnHost) {
                    if constexpr (_OutputOnHost) {
                        element_t* dInputs;
                        ThrowIfBad(cudaMalloc(&dInputs, sizeof(element_t) * _TFixedMLPL::inputCount));
                        ThrowIfBad(cudaMemcpy(dInputs, Inputs, sizeof(element_t) * _TFixedMLPL::inputCount, cudaMemcpyHostToDevice));
                        element_t* dOutputs;
                        ThrowIfBad(cudaMalloc(&dOutputs, sizeof(element_t) * _TFixedMLPL::outputCount));

                        FixedMLPL_Run<_TFixedMLPL, false, false>(MLPL, dInputs, dOutputs);
                    }
                    else {
                        element_t* dInputs;
                        ThrowIfBad(cudaMalloc(&dInputs, sizeof(element_t) * _TFixedMLPL::inputCount));
                        ThrowIfBad(cudaMemcpy(dInputs, Inputs, sizeof(element_t) * _TFixedMLPL::inputCount, cudaMemcpyHostToDevice));

                        FixedMLPL_Run<_TFixedMLPL, false, false>(MLPL, dInputs, Outputs);
                    }
                }
                else {
                    if constexpr (_OutputOnHost) {
                        element_t* dOutputs;
                        ThrowIfBad(cudaMalloc(&dOutputs, sizeof(element_t) * _TFixedMLPL::outputCount));

                        FixedMLPL_Run<_TFixedMLPL, false, false>(MLPL, Inputs, dOutputs);
                    }
                    else {
                        ThrowIfBad(cudaMemcpy(Outputs, &MLPL->bias, sizeof(element_t) * _TFixedMLPL::outputCount, cudaMemcpyDeviceToDevice));

                        cublasHandle_t cublasH;
                        ThrowIfBad(cublasCreate(&cublasH));

                        float oneF = 1.f;
                        ThrowIfBad(cublasSgemv(cublasH, CUBLAS_OP_N, _TFixedMLPL::outputCount, _TFixedMLPL::inputCount, &oneF, (const float*)&MLPL->weights, _TFixedMLPL::outputCount, Inputs, 1, &oneF, Outputs, 1));

                        ThrowIfBad(cublasDestroy(cublasH));
                    }
                }
            }
            template <IsFixedMLP _TFixedMLP, bool _InputOnHost, bool _OutputOnHost>
            inline static void FixedMLP_Run(const _TFixedMLP* MLP, const typename _TFixedMLP::element_t* Inputs, typename _TFixedMLP::element_t* Outputs) {
                using element_t = typename _TFixedMLP::element_t;
                if constexpr (_InputOnHost) {
                    if constexpr (_OutputOnHost) {
                        element_t* dInputs;
                        ThrowIfBad(cudaMalloc(&dInputs, sizeof(element_t) * _TFixedMLP::InputCount()));
                        ThrowIfBad(cudaMemcpy(dInputs, Inputs, sizeof(element_t) * _TFixedMLP::InputCount(), cudaMemcpyHostToDevice));
                        element_t* dOutputs;
                        ThrowIfBad(cudaMalloc(&dOutputs, sizeof(element_t) * _TFixedMLP::OutputCount()));

                        FixedMLP_Run<_TFixedMLP, false, false>(MLP, dInputs, dOutputs);
                    }
                    else {
                        element_t* dInputs;
                        ThrowIfBad(cudaMalloc(&dInputs, sizeof(element_t) * _TFixedMLP::InputCount()));
                        ThrowIfBad(cudaMemcpy(dInputs, Inputs, sizeof(element_t) * _TFixedMLP::InputCount(), cudaMemcpyHostToDevice));

                        FixedMLP_Run<_TFixedMLP, false, false>(MLP, dInputs, Outputs);
                    }
                }
                else {
                    if constexpr (_OutputOnHost) {
                        element_t* dOutputs;
                        ThrowIfBad(cudaMalloc(&dOutputs, sizeof(element_t) * _TFixedMLP::OutputCount()));

                        FixedMLP_Run<_TFixedMLP, false, false>(MLP, Inputs, dOutputs);
                    }
                    else {
                        if constexpr (_TFixedMLP::LayerCount() == 1) {
                            FixedMLPL_Run<decltype(MLP->layer), false, false>(&MLP->layer, Inputs, Outputs);
                        }
                        else {
                            element_t* dIntermediate0;
                            ThrowIfBad(cudaMalloc(&dIntermediate0, sizeof(element_t) * _TFixedMLP::OutputCount()));
                            element_t* dIntermediate1;
                            ThrowIfBad(cudaMalloc(&dIntermediate1, sizeof(element_t) * _TFixedMLP::OutputCount()));

                            details::FixedMLP_Run<_TFixedMLP>(MLP, Inputs, dIntermediate0, dIntermediate1, Outputs);
                        }
                    }
                }
            }

            template <IsFixedMLPL _TFixedMLPL>
            inline static void FixedMLPL_FillWith0(_TFixedMLPL* MLPL) {
                using element_t = typename _TFixedMLPL::element_t;

                rand::ClearArray<false, element_t>(FixedMLPL_GetElementSpan(MLPL));
            }
            template <IsFixedMLPL _TFixedMLPL, std::uniform_random_bit_generator _TRNG>
            inline static void FixedMLPL_FillWithRandom(_TFixedMLPL* MLPL, _TRNG& RNG) {
                using element_t = typename _TFixedMLPL::element_t;

                rand::InitRandomArray<false, element_t, _TRNG>(FixedMLPL_GetElementSpan(MLPL), RNG);
            }
            template <IsFixedMLPL _TFixedMLPL, std::uniform_random_bit_generator _TRNG>
            inline static void FixedMLPL_ChangeWithRandom(_TFixedMLPL* MLPL, typename _TFixedMLPL::element_t Scalar, _TRNG& RNG) {
                using element_t = typename _TFixedMLPL::element_t;

                rand::RandomizeArray<false, element_t, _TRNG>(FixedMLPL_GetElementSpan(MLPL), Scalar, RNG);
            }
            template <IsFixedMLPL _TFixedMLPL, std::uniform_random_bit_generator _TRNG>
            inline static void FixedMLPL_ChangeWithRandom(_TFixedMLPL* MLPL, typename _TFixedMLPL::element_t Scalar, typename _TFixedMLPL::element_t LowerBound, typename _TFixedMLPL::element_t UpperBound, _TRNG& RNG) {
                using element_t = typename _TFixedMLPL::element_t;

                rand::RandomizeArray<false, element_t, _TRNG>(FixedMLPL_GetElementSpan(MLPL), Scalar, LowerBound, UpperBound, RNG);
            }

            template <IsFixedMLP _TFixedMLP>
            inline static void FixedMLP_FillWith0(_TFixedMLP* MLP) {
                using element_t = typename _TFixedMLP::element_t;

                rand::ClearArray<false, element_t>(FixedMLP_GetElementSpan(MLP));
            }
            template <IsFixedMLP _TFixedMLP, std::uniform_random_bit_generator _TRNG>
            inline static void FixedMLP_FillWithRandom(_TFixedMLP* MLP, _TRNG& RNG) {
                using element_t = typename _TFixedMLP::element_t;

                rand::InitRandomArray<false, element_t, _TRNG>(FixedMLP_GetElementSpan(MLP), RNG);
            }
            template <IsFixedMLP _TFixedMLP, std::uniform_random_bit_generator _TRNG>
            inline static void FixedMLP_ChangeWithRandom(_TFixedMLP* MLP, typename _TFixedMLP::element_t Scalar, _TRNG& RNG) {
                using element_t = typename _TFixedMLP::element_t;

                rand::RandomizeArray<false, element_t, _TRNG>(FixedMLP_GetElementSpan(MLP), Scalar, RNG);
            }
            template <IsFixedMLP _TFixedMLP, std::uniform_random_bit_generator _TRNG>
            inline static void FixedMLP_ChangeWithRandom(_TFixedMLP* MLP, typename _TFixedMLP::element_t Scalar, typename _TFixedMLP::element_t LowerBound, typename _TFixedMLP::element_t UpperBound, _TRNG& RNG) {
                using element_t = typename _TFixedMLP::element_t;

                rand::RandomizeArray<false, element_t, _TRNG>(FixedMLP_GetElementSpan(MLP), Scalar, LowerBound, UpperBound, RNG);
            }
        }
    }
}