#pragma once

#include "ai_evol_eval_output.h"
#include "fields_dfield.h"
#include "rand_anyrng.h"
#include <tuple>

namespace bcuda {
    namespace details {
        template <typename _T, size_t _DimensionCount>
        struct FieldInstance_CurrentInstance final {
            fields::DField<_T, _DimensionCount> dfield;
            ArrayV<size_t> inputs;
            ArrayV<size_t> outputs;
            void* obj;
            void* objectRunner_sharedData;
            rand::AnyRNG<uint32_t> rng;
        };
    }
    namespace fields {
        template <typename _T, size_t _DimensionCount>
        using fieldInstance_objectRunner_t = void(*)(void* Object, fields::DFieldProxy<_T, _DimensionCount> Field, void* SharedData);
        template <typename _T, size_t _DimensionCount>
        using fieldInstance_createField_t = DField<_T, _DimensionCount>*(*)(void* SharedData);
        struct FieldInstance_Construct_Settings final {
            rand::AnyRNG<uint32_t> rng;
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
        void* FieldInstance_Construct(void* Object, void* Settings) {
            FieldInstance_Construct_Settings settings = *(FieldInstance_Construct_Settings*)Settings;
            rand::AnyRNG<uint32_t>& rng = settings.rng;

            details::FieldInstance_CurrentInstance<_T, _DimensionCount>* p_rv = new details::FieldInstance_CurrentInstance<_T, _DimensionCount>{ _CreateField(settings.createField_sharedData), ArrayV<uint32_3>(settings.inputCount), ArrayV<uint32_3>(settings.outputCount), Object, settings.objectRunner_sharedData, rng };
            details::FieldInstance_CurrentInstance<_T, _DimensionCount>& rv = *p_rv;
            ArrayV<size_t>& il = rv.inputs;
            ArrayV<size_t>& ol = rv.outputs;

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
        }
        template <typename _T, size_t _DimensionCount, fieldInstance_objectRunner_t<_T, _DimensionCount> _ObjectRunner>
        _T* FieldInstance_Iterate(void* CurrentInstance, _T* Inputs) {
            details::FieldInstance_CurrentInstance<_T, _DimensionCount>& c = *(details::FieldInstance_CurrentInstance<_T, _DimensionCount>*)CurrentInstance;
            DField<_T, _DimensionCount>& df = c.dfield;
            ArrayV<size_t>& il = c.inputs;
            ArrayV<size_t>& ol = c.outputs;

            if (Inputs) {
                FieldProxy<_T, _DimensionCount> f = df.F();
                for (size_t i = 0; i < il.size; ++i) {
                    f.CpyValIn(il[i], Inputs[i]);
                }
            }
            _ObjectRunner(c.obj, df.MakeProxy(), c.objectRunner_sharedData);
            df.Reverse();
            _T* opts = new _T[ol.size];
            {
                FieldProxy<_T, _DimensionCount> f = df.F();
                for (size_t i = 0; i < ol.size; ++i) {
                    opts[i] = f.CpyValOut(ol[i]);
                }
            }
            return opts;
        }
        template <typename _T, size_t _DimensionCount>
        void FieldInstance_Destruct(void* CurrentInstance) {
            delete (details::FieldInstance_CurrentInstance<_T, _DimensionCount>*)CurrentInstance;
        }
        template <typename _T, size_t _DimensionCount, fieldInstance_createField_t<_T, _DimensionCount> _CreateField, fieldInstance_objectRunner_t<_T, _DimensionCount> _ObjectRunner>
        ai::evol::eval::output::InstanceFunctions<_T*, _T*> FieldInstance() {
            ai::evol::eval::output::InstanceFunctions<_T*, _T*> ifs;
            ifs.constructInstance = FieldInstance_Construct<_T, _DimensionCount, _CreateField>;
            ifs.iterateInstance = FieldInstance_Iterate<_T, _DimensionCount, _ObjectRunner>;
            ifs.destructInstance = FieldInstance_Destruct<_T, _DimensionCount>;
            return ifs;
        }

        template <typename _TFieldValue, size_t _DimensionCount, typename _TInput, typename _TOutput, fieldInstance_assignInput_t<_TFieldValue, _TInput> _AssignInput, fieldInstance_getOutput_t<_TFieldValue, _TOutput> _GetOutput, fieldInstance_objectRunner_t<_TFieldValue, _DimensionCount> _ObjectRunner>
        _TOutput* FieldInstance_Iterate(void* CurrentInstance, _TInput* Inputs) {
            details::FieldInstance_CurrentInstance<_TFieldValue, _DimensionCount> c = *(details::FieldInstance_CurrentInstance<_TFieldValue, _DimensionCount>*)CurrentInstance;
            DField<_TFieldValue, _DimensionCount>& df = c.dfield;
            ArrayV<size_t>& il = c.inputs;
            ArrayV<size_t>& ol = c.outputs;

            if (Inputs) {
                FieldProxy<_TFieldValue, _DimensionCount> f = df.F();
                for (size_t i = 0; i < il.size; ++i) {
                    size_t idx = il[i];
                    _TFieldValue fieldValue = f.CpyValOut(idx);
                    _AssignInput(fieldValue, Inputs[i]);
                    f.CpyValIn(idx, fieldValue);
                }
            }
            _ObjectRunner(c.obj, df.MakeProxy(), c.objectRunner_sharedData);
            df.Reverse();
            _TOutput* opts = new _TOutput[ol.size];
            {
                FieldProxy<_TFieldValue, _DimensionCount> f = df.F();
                for (size_t i = 0; i < ol.size; ++i) {
                    _TFieldValue fieldValue = f.CpyValOut(ol[i]);
                    opts[i] = _GetOutput(fieldValue);
                }
            }
            return opts;
        }
        template <typename _TFieldValue, size_t _DimensionCount, typename _TInput, typename _TOutput, fieldInstance_createField_t<_TFieldValue, _DimensionCount> _CreateField, fieldInstance_assignInput_t<_TFieldValue, _TInput> _AssignInput, fieldInstance_getOutput_t<_TFieldValue, _TOutput> _GetOutput, fieldInstance_objectRunner_t<_TFieldValue, _DimensionCount> _ObjectRunner>
        ai::evol::eval::output::InstanceFunctions<_TInput*, _TOutput*> FieldInstance() {
            ai::evol::eval::output::InstanceFunctions<_TInput*, _TOutput*> ifs;
            ifs.constructInstance = FieldInstance_Construct<_TFieldValue, _DimensionCount, _CreateField>;
            ifs.iterateInstance = FieldInstance_Iterate<_TFieldValue, _DimensionCount, _TInput, _TOutput, _AssignInput, _GetOutput, _ObjectRunner>;
            ifs.destructInstance = FieldInstance_Destruct<_TFieldValue, _DimensionCount>;
            return ifs;
        }
    }
}