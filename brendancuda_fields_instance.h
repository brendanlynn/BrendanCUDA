#pragma once

#include "brendancuda_fields_dfield.h"
#include "brendancuda_rand_anyrng.h"
#include <tuple>
#include "brendancuda_ai_evol_eval_output.h"

namespace BrendanCUDA {
    namespace details {
        template <typename _T, size_t _DimensionCount>
        struct FieldInstance_CurrentInstance final {
            Fields::DField<_T, _DimensionCount>& field;
            ArrayV<size_t> inputs;
            ArrayV<size_t> outputs;
            void* obj;
            void* objectRunner_sharedData;
            Random::AnyRNG<uint32_t> rng;
        };
    }
    namespace Fields {
        template <typename _T, size_t _DimensionCount>
        using fieldInstance_objectRunner_t = void(*)(void* Object, Fields::DField<_T, _DimensionCount> Field, void* SharedData);
        template <typename _T, size_t _DimensionCount>
        using fieldInstance_createField_t = DField<_T, _DimensionCount>*(*)(void* SharedData);
        struct FieldInstance_Construct_Settings final {
            Random::AnyRNG<uint32_t> rng;
            void* objectRunner_sharedData;
            void* createField_sharedData;
            size_t inputCount;
            size_t outputCount;
        };

        template <typename _TFieldValue, typename _TOutput>
        using fieldInstance_getOutput_t = _TOutput(*)(const _TFieldValue& FieldValue);
        template <typename _TFieldValue, typename _TInput>
        using fieldInstance_assignInput_t = void(*)(_TFieldValue& FieldValue, const _TInput& InputValue);

        template <typename _T, size_t _DimensionCount, fieldInstance_createField_t<_T, _DimensionCount> _CreateField>
        void* FieldInstance_Construct(void* Object, void* Settings);
        template <typename _T, size_t _DimensionCount, fieldInstance_objectRunner_t<_T, _DimensionCount> _ObjectRunner>
        _T* FieldInstance_Iterate(void* CurrentInstance, _T* Inputs);
        template <typename _T, size_t _DimensionCount>
        void FieldInstance_Destruct(void* CurrentInstance);
        template <typename _T, size_t _DimensionCount, fieldInstance_createField_t<_T, _DimensionCount> _CreateField, fieldInstance_objectRunner_t<_T, _DimensionCount> _ObjectRunner>
        AI::Evolution::Evaluation::Output::InstanceFunctions<_T*, _T*> FieldInstance();

        template <typename _TFieldValue, size_t _DimensionCount, typename _TInput, typename _TOutput, fieldInstance_assignInput_t<_TFieldValue, _TInput> _AssignInput, fieldInstance_getOutput_t<_TFieldValue, _TOutput> _GetOutput, fieldInstance_objectRunner_t<_TFieldValue, _DimensionCount> _ObjectRunner>
        _TOutput* FieldInstance_Iterate(void* CurrentInstance, _TInput* Inputs);
        template <typename _TFieldValue, size_t _DimensionCount, typename _TInput, typename _TOutput, fieldInstance_createField_t<_TFieldValue, _DimensionCount> _CreateField, fieldInstance_assignInput_t<_TFieldValue, _TInput> _AssignInput, fieldInstance_getOutput_t<_TFieldValue, _TOutput> _GetOutput, fieldInstance_objectRunner_t<_TFieldValue, _DimensionCount> _ObjectRunner>
        AI::Evolution::Evaluation::Output::InstanceFunctions<_TInput*, _TOutput*> FieldInstance();
    }
}

template <typename _T, size_t _DimensionCount, BrendanCUDA::Fields::fieldInstance_createField_t<_T, _DimensionCount> _CreateField>
void* BrendanCUDA::Fields::FieldInstance_Construct(void* Object, void* Settings) {
    FieldInstance_Construct_Settings settings = *(FieldInstance_Construct_Settings*)Settings;
    Random::AnyRNG<uint32_t>& rng = settings.rng;

    DField<_T, _DimensionCount>& f = *_CreateField(settings.createField_sharedData);
    details::FieldInstance_CurrentInstance<_T, _DimensionCount>* p_rv = new details::FieldInstance_CurrentInstance<_T, _DimensionCount>{ f, ArrayV<uint32_3>(settings.inputCount), ArrayV<uint32_3>(settings.outputCount), Object, settings.objectRunner_sharedData, rng };
    details::FieldInstance_CurrentInstance<_T, _DimensionCount>& rv = *p_rv;
    ArrayV<size_t> il = rv.inputs;
    ArrayV<size_t> ol = rv.outputs;

    std::uniform_int_distribution<uint32_t> dis(0, f.ValueCount());

    for (uint32_t i = 0; i < settings.inputCount; ++i) {
    ContinueInputs:
        size_t thisIdx = dis(rng);
        for (uint32_t j = 0; j < i; ++j)
            if (il[j] == thisIdx)
                goto ContinueInputs;
        il[i] = thisIdx;
    }
    for (uint32_t i = 0; i < settings.outputCount; ++i) {
    ContinueOutputs:
        size_t thisIdx = dis(rng);
        for (uint32_t j = 0; j < i; ++j)
            if (ol[j] == thisIdx)
                goto ContinueOutputs;
        ol[i] = thisIdx;
    }

    return p_rv;
};
template <typename _T, size_t _DimensionCount, BrendanCUDA::Fields::fieldInstance_objectRunner_t<_T, _DimensionCount> _ObjectRunner>
_T* BrendanCUDA::Fields::FieldInstance_Iterate(void* CurrentInstance, _T* Inputs) {
    details::FieldInstance_CurrentInstance<_T, _DimensionCount> c = *(details::FieldInstance_CurrentInstance<_T, _DimensionCount>*)CurrentInstance;
    DField<_T, _DimensionCount>& df = c.field;
    ArrayV<size_t> il = c.inputs;
    ArrayV<size_t> ol = c.outputs;

    if (Inputs) {
        Field<_TFieldValue, _DimensionCount> f = df.FFront();
        for (size_t i = 0; i < il.size; ++i) {
            f.SetValueAt(il[i], Inputs[i]);
        }
    }
    _ObjectRunner(c.obj, df, c.objectRunner_sharedData);
    _T* opts = new _T[ol.size];
    {
        Field<_TFieldValue, _DimensionCount> f = df.FFront();
        for (size_t i = 0; i < ol.size; ++i) {
            opts[i] = f.GetValueAt(ol[i]);
        }
    }
    return opts;
};
template <typename _T, size_t _DimensionCount>
void BrendanCUDA::Fields::FieldInstance_Destruct(void* CurrentInstance) {
    details::FieldInstance_CurrentInstance<_T, _DimensionCount>* p_c = (details::FieldInstance_CurrentInstance<_T, _DimensionCount>*)CurrentInstance;
    details::FieldInstance_CurrentInstance<_T, _DimensionCount> c = *p_c;
    DField<_T, _DimensionCount>& f = c.field;

    f.Dispose();
    delete (&f);
    c.inputs.Dispose();
    c.outputs.Dispose();
    delete p_c;
};

template <typename _T, size_t _DimensionCount, BrendanCUDA::Fields::fieldInstance_createField_t<_T, _DimensionCount> _CreateField, BrendanCUDA::Fields::fieldInstance_objectRunner_t<_T, _DimensionCount> _ObjectRunner>
BrendanCUDA::AI::Evolution::Evaluation::Output::InstanceFunctions<_T*, _T*> BrendanCUDA::Fields::FieldInstance() {
    AI::Evolution::Evaluation::Output::InstanceFunctions<_T*, _T*> ifs;
    ifs.constructInstance = FieldInstance_Construct<_T, _DimensionCount, _CreateField>;
    ifs.iterateInstance = FieldInstance_Iterate<_T, _DimensionCount, _ObjectRunner>;
    ifs.destructInstance = FieldInstance_Destruct<_T, _DimensionCount>;
    return ifs;
}

template <typename _TFieldValue, size_t _DimensionCount, typename _TInput, typename _TOutput, BrendanCUDA::Fields::fieldInstance_assignInput_t<_TFieldValue, _TInput> _AssignInput, BrendanCUDA::Fields::fieldInstance_getOutput_t<_TFieldValue, _TOutput> _GetOutput, BrendanCUDA::Fields::fieldInstance_objectRunner_t<_TFieldValue, _DimensionCount> _ObjectRunner>
_TOutput* BrendanCUDA::Fields::FieldInstance_Iterate(void* CurrentInstance, _TInput* Inputs) {
    details::FieldInstance_CurrentInstance<_TFieldValue, _DimensionCount> c = *(details::FieldInstance_CurrentInstance<_TFieldValue, _DimensionCount>*)CurrentInstance;
    DField<_TFieldValue, _DimensionCount>& df = c.field;
    ArrayV<size_t> il = c.inputs;
    ArrayV<size_t> ol = c.outputs;

    if (Inputs) {
        Field<_TFieldValue, _DimensionCount> f = df.FFront();
        for (size_t i = 0; i < il.size; ++i) {
            size_t idx = il[i];
            _TFieldValue fieldValue = f.GetValueAt(idx);
            _AssignInput(fieldValue, Inputs[i]);
            f.SetValueAt(idx, fieldValue);
        }
    }
    _ObjectRunner(c.obj, df, c.objectRunner_sharedData);
    _TOutput* opts = new _TOutput[ol.size];
    {
        Field<_TFieldValue, _DimensionCount> f = df.FFront();
        for (size_t i = 0; i < ol.size; ++i) {
            _TFieldValue fieldValue = f.GetValueAt(ol[i]);
            opts[i] = _GetOutput(fieldValue);
        }
    }
    return opts;
}

template <typename _TFieldValue, size_t _DimensionCount, typename _TInput, typename _TOutput, BrendanCUDA::Fields::fieldInstance_createField_t<_TFieldValue, _DimensionCount> _CreateField, BrendanCUDA::Fields::fieldInstance_assignInput_t<_TFieldValue, _TInput> _AssignInput, BrendanCUDA::Fields::fieldInstance_getOutput_t<_TFieldValue, _TOutput> _GetOutput, BrendanCUDA::Fields::fieldInstance_objectRunner_t<_TFieldValue, _DimensionCount> _ObjectRunner>
BrendanCUDA::AI::Evolution::Evaluation::Output::InstanceFunctions<_TInput*, _TOutput*> BrendanCUDA::Fields::FieldInstance() {
    AI::Evolution::Evaluation::Output::InstanceFunctions<_TInput*, _TOutput*> ifs;
    ifs.constructInstance = FieldInstance_Construct<_TFieldValue, _DimensionCount, _CreateField>;
    ifs.iterateInstance = FieldInstance_Iterate<_TFieldValue, _DimensionCount, _TInput, _TOutput, _AssignInput, _GetOutput, _ObjectRunner>;
    ifs.destructInstance = FieldInstance_Destruct<_TFieldValue, _DimensionCount>;
    return ifs;
}