#pragma once

#include "brendancuda_fields_dfield3.h"
#include "brendancuda_rand_anyrng.h"
#include <tuple>
#include "brendancuda_rand_sseed.h"
#include "brendancuda_ai_evol_eval_output.h"

namespace BrendanCUDA {
    namespace details {
        template <typename _T>
        struct FieldInstance3_CurrentInstance final {
            Fields::DField3<_T>& field;
            ArrayV<uint32_3> inputs;
            ArrayV<uint32_3> outputs;
            void* obj;
            void* objectRunner_sharedData;
            Random::AnyRNG<uint32_t> rng;
        };
    }
    namespace Fields {
        template <typename _T>
        using fieldInstance3_objectRunner_t = void(*)(void* Object, DField3<_T> Field, void* SharedData);
        template <typename _T>
        using fieldInstance3_createField_t = DField3<_T>*(*)(void* SharedData);
        struct FieldInstance3_Construct_Settings final {
            Random::AnyRNG<uint32_t> rng;
            void* objectRunner_sharedData;
            void* createField_sharedData;
            size_t inputCount;
            size_t outputCount;
        };

        template <typename _TFieldValue, typename _TOutput>
        using fieldInstance3_getOutput_t = _TOutput(*)(const _TFieldValue& FieldValue);
        template <typename _TFieldValue, typename _TInput>
        using fieldInstance3_assignInput_t = void(*)(_TFieldValue& FieldValue, const _TInput& InputValue);

        template <typename _T, fieldInstance3_createField_t<_T> _CreateField>
        void* FieldInstance3_Construct(void* Object, void* Settings);
        template <typename _T, fieldInstance3_objectRunner_t<_T> _ObjectRunner>
        _T* FieldInstance3_Iterate(void* CurrentInstance, _T* Inputs);
        template <typename _T>
        void FieldInstance3_Destruct(void* CurrentInstance);
        template <typename _T, fieldInstance3_createField_t<_T> _CreateField, fieldInstance3_objectRunner_t<_T> _ObjectRunner>
        AI::Evolution::Evaluation::Output::InstanceFunctions<_T*, _T*> FieldInstance3();

        template <typename _TFieldValue, typename _TInput, typename _TOutput, fieldInstance3_assignInput_t<_TFieldValue, _TInput> _AssignInput, fieldInstance3_getOutput_t<_TFieldValue, _TOutput> _GetOutput, fieldInstance3_objectRunner_t<_TFieldValue> _ObjectRunner>
        _TOutput* FieldInstance3_Iterate(void* CurrentInstance, _TInput* Inputs);
        template <typename _TFieldValue, typename _TInput, typename _TOutput, fieldInstance3_createField_t<_TFieldValue> _CreateField, fieldInstance3_assignInput_t<_TFieldValue, _TInput> _AssignInput, fieldInstance3_getOutput_t<_TFieldValue, _TOutput> _GetOutput, fieldInstance3_objectRunner_t<_TFieldValue> _ObjectRunner>
        AI::Evolution::Evaluation::Output::InstanceFunctions<_TInput*, _TOutput*> FieldInstance3();
    }
}

template <typename _T, BrendanCUDA::Fields::fieldInstance3_createField_t<_T> _CreateField>
void* BrendanCUDA::Fields::FieldInstance3_Construct(void* Object, void* Settings) {
    FieldInstance3_Construct_Settings settings = *(FieldInstance3_Construct_Settings*)Settings;
    Random::AnyRNG<uint32_t>& rng = settings.rng;

    DField3<_T>& f = *_CreateField(settings.createField_sharedData);
    details::FieldInstance3_CurrentInstance<_T>* p_rv = new details::FieldInstance3_CurrentInstance<_T>{ f, ArrayV<uint32_3>(settings.inputCount), ArrayV<uint32_3>(settings.outputCount), Object, settings.objectRunner_sharedData, rng };
    details::FieldInstance3_CurrentInstance<_T>& rv = *p_rv;
    ArrayV<uint32_3> il = rv.inputs;
    ArrayV<uint32_3> ol = rv.outputs;

    std::uniform_int_distribution<uint64_t> disU64(0, std::numeric_limits<uint64_t>::max());

    std::uniform_int_distribution<uint32_t> disX(0, f.LengthX() - 1);
    std::uniform_int_distribution<uint32_t> disY(0, f.LengthY() - 1);
    std::uniform_int_distribution<uint32_t> disZ(0, f.LengthZ() - 1);

    for (size_t i = 0; i < settings.inputCount; ++i) {
        il[i] = uint32_3{ disX(rng), disY(rng), disZ(rng) };
    }

    for (size_t i = 0; i < settings.outputCount; ++i) {
        ol[i] = uint32_3{ disX(rng), disY(rng), disZ(rng) };
    }

    return p_rv;
};
template <typename _T, BrendanCUDA::Fields::fieldInstance3_objectRunner_t<_T> _ObjectRunner>
_T* BrendanCUDA::Fields::FieldInstance3_Iterate(void* CurrentInstance, _T* Inputs) {
    details::FieldInstance3_CurrentInstance<_T> c = *(details::FieldInstance3_CurrentInstance<_T>*)CurrentInstance;
    DField3<_T>& f = c.field;
    ArrayV<uint32_3> il = c.inputs;
    ArrayV<uint32_3> ol = c.outputs;

    if (Inputs) {
        for (size_t i = 0; i < il.size; ++i) {
            f.FFront().SetValueAt(il[i], Inputs[i]);
        }
    }
    _ObjectRunner(c.obj, f, c.objectRunner_sharedData);
    _T* opts = new _T[ol.size];
    for (size_t i = 0; i < ol.size; ++i) {
        opts[i] = f.FFront().GetValueAt(ol[i]);
    }
    return opts;
};
template <typename _T>
void BrendanCUDA::Fields::FieldInstance3_Destruct(void* CurrentInstance) {
    details::FieldInstance3_CurrentInstance<_T>* p_c = (details::FieldInstance3_CurrentInstance<_T>*)CurrentInstance;
    details::FieldInstance3_CurrentInstance<_T> c = *p_c;
    DField3<_T>& f = c.field;

    f.Dispose();
    delete (&f);
    c.inputs.Dispose();
    c.outputs.Dispose();
    delete p_c;
};

template <typename _T, BrendanCUDA::Fields::fieldInstance3_createField_t<_T> _CreateField, BrendanCUDA::Fields::fieldInstance3_objectRunner_t<_T> _ObjectRunner>
BrendanCUDA::AI::Evolution::Evaluation::Output::InstanceFunctions<_T*, _T*> BrendanCUDA::Fields::FieldInstance3() {
    AI::Evolution::Evaluation::Output::InstanceFunctions<_T*, _T*> ifs;
    ifs.constructInstance = FieldInstance3_Construct<_T, _CreateField>;
    ifs.iterateInstance = FieldInstance3_Iterate<_T, _ObjectRunner>;
    ifs.destructInstance = FieldInstance3_Destruct<_T>;
    return ifs;
}

template <typename _TFieldValue, typename _TInput, typename _TOutput, BrendanCUDA::Fields::fieldInstance3_assignInput_t<_TFieldValue, _TInput> _AssignInput, BrendanCUDA::Fields::fieldInstance3_getOutput_t<_TFieldValue, _TOutput> _GetOutput, BrendanCUDA::Fields::fieldInstance3_objectRunner_t<_TFieldValue> _ObjectRunner>
_TOutput* BrendanCUDA::Fields::FieldInstance3_Iterate(void* CurrentInstance, _TInput* Inputs) {
    details::FieldInstance3_CurrentInstance<_TFieldValue> c = *(details::FieldInstance3_CurrentInstance<_TFieldValue>*)CurrentInstance;
    DField3<_TFieldValue>& f = c.field;
    ArrayV<uint32_3> il = c.inputs;
    ArrayV<uint32_3> ol = c.outputs;

    if (Inputs) {
        for (size_t i = 0; i < il.size; ++i) {
            auto& idx = il[i];
            _TFieldValue fieldValue = f.FFront().GetValueAt(idx);
            _AssignInput(fieldValue, Inputs[i]);
            f.FFront().SetValueAt(idx, fieldValue);
        }
    }
    _ObjectRunner(c.obj, f, c.objectRunner_sharedData);
    _TOutput* opts = new _TOutput[ol.size];
    for (size_t i = 0; i < ol.size; ++i) {
        _TFieldValue fieldValue = f.FFront().GetValueAt(ol[i]);
        opts[i] = _GetOutput(fieldValue);
    }
    return opts;
}

template <typename _TFieldValue, typename _TInput, typename _TOutput, BrendanCUDA::Fields::fieldInstance3_createField_t<_TFieldValue> _CreateField, BrendanCUDA::Fields::fieldInstance3_assignInput_t<_TFieldValue, _TInput> _AssignInput, BrendanCUDA::Fields::fieldInstance3_getOutput_t<_TFieldValue, _TOutput> _GetOutput, BrendanCUDA::Fields::fieldInstance3_objectRunner_t<_TFieldValue> _ObjectRunner>
BrendanCUDA::AI::Evolution::Evaluation::Output::InstanceFunctions<_TInput*, _TOutput*> BrendanCUDA::Fields::FieldInstance3() {
    AI::Evolution::Evaluation::Output::InstanceFunctions<_TInput*, _TOutput*> ifs;
    ifs.constructInstance = FieldInstance3_Construct<_TFieldValue, _CreateField>;
    ifs.iterateInstance = FieldInstance3_Iterate<_TFieldValue, _TInput, _TOutput, _AssignInput, _GetOutput, _ObjectRunner>;
    ifs.destructInstance = FieldInstance3_Destruct<_TFieldValue>;
    return ifs;
}