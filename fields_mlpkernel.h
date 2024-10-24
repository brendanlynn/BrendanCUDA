#pragma once
#pragma once

#include "fields_dfield.h"
#include "ai_mlp_fixedmlp.h"

namespace bcuda {
    namespace fields {
        namespace details {
            template <typename _TMLP, size_t _Index = 0>
            __forceinline void WriteHiddenWidthsToArray(size_t* WidthsArray) {
                if (_Index + 1 >= _TMLP::LayerCount()) return;

                WidthsArray[_Index] = _TMLP::template widthAt<_Index + 1>;
                WriteOutputWidthsToArray<_TMLP, _Index + 1>(WidthsArray);
            }

            static inline constexpr size_t AreaCountByDimension(size_t Dimension) {
                size_t total = 3;
                for (size_t i = 1; i < Dimension; ++i)
                    total *= 3;
                return total;
            }

            template <std::floating_point _TFloat>
            void RunKernelMLPOverDField(Span<const size_t> Dimensions, _TFloat* FieldFData, _TFloat* FieldBData, size_t FieldElementSize, _TFloat* KernelMLP, Span<const size_t> HiddenWidths, ai::activationFunction_t<_TFloat> ActivationFunc, ptrdiff_t InputPrivateDiff, size_t InputPrivateCount, ptrdiff_t InputSharedDiff, size_t InputSharedCount, ptrdiff_t OutputDiff, size_t OutputCount);
        }

        template <std::floating_point _TFloat, typename _TFieldVal, size_t _DimensionCount, ai::mlp::IsFixedMLP _TMLP>
        void RunKernelMLPOverDFieldProxy(fields::DFieldProxy<_TFieldVal, _DimensionCount> DField, _TMLP* KernelMLP, ptrdiff_t InputPrivateDiff, size_t InputPrivateCount, ptrdiff_t InputSharedDiff, size_t InputSharedCount, ptrdiff_t OutputDiff, size_t OutputCount) {
            static_assert(std::same_as<typename _TMLP::element_t, _TFloat>, "_TMLP is incompatible.");
            
            constexpr size_t areaCountByDim = details::AreaCountByDimension(_DimensionCount);
            size_t inputCount = InputPrivateCount + areaCountByDim * InputSharedCount;

            if (_TMLP::InputCount() == inputCount) throw std::exception("_TMLP must have an input count of InputThisDiff + (3 ** _DimensionCount - 1) * InputSharedDiff!");
            if (_TMLP::OutputCount() == OutputCount) throw std::exception("_TMLP must have an output count of OutputCount!");

            FixedVector<uint32_t, _DimensionCount> dimensions = DField.Dimensions();
            Span<const size_t> dimensionsSpan(dimensions.v, _DimensionCount);

            size_t hiddenWidths[_TMLP::LayerCount() - 1];
            details::WriteHiddenWidthsToArray<_TMLP>(hiddenWidths);
            Span<const size_t> hiddenWidthsSpan(hiddenWidths, _TMLP::LayerCount());

            details::RunKernelMLPOverDField<_TFloat>(dimensionsSpan, (_TFloat*)DField.FData(), (_TFloat*)DField.BData(), sizeof(_TFieldVal), (_TFloat*)KernelMLP, hiddenWidthsSpan, _TMLP::activationFunction, InputPrivateDiff, InputPrivateCount, InputSharedDiff, InputSharedCount, OutputDiff, OutputCount);
        }
        template <std::floating_point _TFloat, typename _TFieldVal, size_t _DimensionCount, ai::mlp::IsFixedMLP _TMLP>
        void RunKernelMLPOverDField(fields::DField<_TFieldVal, _DimensionCount>& DField, _TMLP* KernelMLP, ptrdiff_t InputPrivateDiff, size_t InputPrivateCount, ptrdiff_t InputSharedDiff, size_t InputSharedCount, ptrdiff_t OutputDiff, size_t OutputCount) {
            RunKernelMLPOverDFieldProxy<_TFloat, _TFieldVal, _DimensionCount, _TMLP>(DField, KernelMLP, InputPrivateDiff, InputPrivateCount, InputSharedDiff, InputSharedCount, OutputDiff, OutputCount);
            DField.Reverse();
        }
    }
}