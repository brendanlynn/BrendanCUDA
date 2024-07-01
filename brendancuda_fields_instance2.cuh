#pragma once

#include "brendancuda_fields_dfield2.cuh"
#include "brendancuda_random_anyrng.cuh"
#include <tuple>
#include "brendancuda_random_sseed.cuh"
#include "brendancuda_ai_evolution_evaluation_output.h"

namespace BrendanCUDA {
    namespace details {
        template <typename _T>
        struct FieldInstance2_CurrentInstance final {
            Fields::DField2<_T>& field;
            uint32_2* inputs;
            size_t inputCount;
            uint32_2* outputs;
            size_t outputCount;
            void* obj;
            void* objectRunner_sharedData;
            Random::AnyRNG<uint32_t> rng;
        };
    }
    namespace Fields {
        template <typename _T>
        using fieldInstance2_objectRunner_t = void(*)(void* Object, DField2<_T> Field, void* SharedData);
        template <typename _T>
        using fieldInstance2_createField_t = DField2<_T>*(*)(void* SharedData);
        struct FieldInstance2_Construct_Settings final {
            Random::AnyRNG<uint32_t> rng;
            void* objectRunner_sharedData;
            void* createField_sharedData;
            size_t inputCount;
            size_t outputCount;
        };
        template <typename _T, fieldInstance2_createField_t<_T> _CreateField>
        void* FieldInstance2_Construct(void* Object, void* Settings);
        template <typename _T, fieldInstance2_objectRunner_t<_T> _ObjectRunner>
        _T* FieldInstance2_Iterate(void* CurrentInstance, _T* Inputs);
        template <typename _T>
        void FieldInstance2_Destruct(void* CurrentInstance);
        template <typename _T, fieldInstance2_createField_t<_T> _CreateField, fieldInstance2_objectRunner_t<_T> _ObjectRunner>
        constexpr AI::Evolution::Evaluation::Output::InstanceFunctions<_T> FieldInstance2();
    }
}

template <typename _T, BrendanCUDA::Fields::fieldInstance2_createField_t<_T> _CreateField>
void* BrendanCUDA::Fields::FieldInstance2_Construct(void* Object, void* Settings) {
    FieldInstance2_Construct_Settings settings = *(FieldInstance2_Construct_Settings*)Settings;
    Random::AnyRNG<uint32_t>& rng = settings.rng;

    DField2<_T>& f = *_CreateField(settings.createField_sharedData);
    details::FieldInstance2_CurrentInstance<_T>* p_rv = new details::FieldInstance2_CurrentInstance<_T>{ f, 0, settings.inputCount, 0, settings.outputCount, Object, settings.objectRunner_sharedData, rng };
    details::FieldInstance2_CurrentInstance<_T>& rv = *p_rv;
    uint32_2*& il = rv.inputs;
    uint32_2*& ol = rv.outputs;

    std::uniform_int_distribution<uint64_t> disU64(0, std::numeric_limits<uint64_t>::max());

    std::uniform_int_distribution<uint32_t> disX(0, f.LengthX() - 1);
    std::uniform_int_distribution<uint32_t> disY(0, f.LengthY() - 1);

    il = new uint32_2[settings.inputCount];
    for (size_t i = 0; i < settings.inputCount; ++i) {
        il[i] = uint32_2{ disX(rng), disY(rng) };
    }

    ol = new uint32_2[settings.outputCount];
    for (size_t i = 0; i < settings.outputCount; ++i) {
        ol[i] = uint32_2{ disX(rng), disY(rng) };
    }

    return p_rv;
};
template <typename _T, BrendanCUDA::Fields::fieldInstance2_objectRunner_t<_T> _ObjectRunner>
_T* BrendanCUDA::Fields::FieldInstance2_Iterate(void* CurrentInstance, _T* Inputs) {
    details::FieldInstance2_CurrentInstance<_T> c = *(details::FieldInstance2_CurrentInstance<_T>*)CurrentInstance;
    DField2<_T>& f = c.field;
    uint32_2* il = c.inputs;
    uint32_2* ol = c.outputs;

    if (Inputs) {
        for (size_t i = 0; i < fieldInputLength; ++i) {
            f.FFront().SetValueAt(il[i], Inputs[i]);
        }
    }
    _ObjectRunner(c.obj, f, c.objectRunner_sharedData);
    _T* opts = new _T[c.outputCount];
    for (size_t i = 0; i < c.outputCount; ++i) {
        opts[i] = f.FFront().GetValueAt(ol[i]);
    }
    return opts;
};
template <typename _T>
void BrendanCUDA::Fields::FieldInstance2_Destruct(void* CurrentInstance) {
    details::FieldInstance2_CurrentInstance<_T>* p_c = (details::FieldInstance2_CurrentInstance<_T>*)CurrentInstance;
    details::FieldInstance2_CurrentInstance<_T> c = *p_c;
    DField2<_T>& f = c.field;

    f.Dispose();
    delete (&f);
    delete[] c.inputs;
    delete[] c.outputs;
    delete p_c;
};

template <typename _T, BrendanCUDA::Fields::fieldInstance2_createField_t<_T> _CreateField, BrendanCUDA::Fields::fieldInstance2_objectRunner_t<_T> _ObjectRunner>
constexpr BrendanCUDA::AI::Evolution::Evaluation::Output::InstanceFunctions<_T> BrendanCUDA::Fields::FieldInstance2() {
    AI::Evolution::Evaluation::Output::InstanceFunctions<_T> ifs;
    ifs.constructInstance = FieldInstance2_Construct<_T, _CreateField>;
    ifs.iterateInstance = FieldInstance2_Iterate<_T, _ObjectRunner>;
    ifs.destructInstance = FieldInstance2_Destruct<_T>;
    return ifs;
}