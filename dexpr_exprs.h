#pragma once

#include <unordered_map>
#include <cmath>

namespace BrendanCUDA {
    namespace DExpr {
        template <typename _TOutput>
        struct Expr {
            virtual _TOutput Calc(const std::unordered_map<uint64_t, void*>&);
        };

        template <typename _T, _T _Val>
        struct Val : public Expr<_T> {
            _T Calc(const std::unordered_map<uint64_t, void*>&) {
                return _Val;
            }
        };
        template <typename _T>
        struct Val : public Expr<_T> {
            _T val;

            _T Calc(const std::unordered_map<uint64_t, void*>&) {
                return val;
            }
        };
        template <typename _T, uint64_t _Idx>
        struct Var : public Expr<_T> {
            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                return *(_T*)Map(_Idx);
            }
        };
        template <typename _T>
        struct Var : public Expr<_T> {
            uint64_t idx;

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                return *(_T*)Map(idx);
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            Expr<_T>* v[_Count];

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                _T t = _T{};
                for (size_t i = 0; i < _Count; ++i)
                    t += v[i]->Calc(Map);
                return t;
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            size_t count;
            Expr<_T>** v;

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                _T t = _T{};
                for (size_t i = 0; i < count; ++i)
                    t += v[i]->Calc(Map);
                return t;
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Subtract : public Expr<_T> {
            Expr<_T>* a;
            Expr<_T>* b;

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                return a->Calc(Map) - b->Calc(Map);
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            Expr<_T>* v[_Count];

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                _T t = (_T)1;
                for (size_t i = 0; i < _Count; ++i)
                    t *= v[i]->Calc(Map);
                return t;
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            size_t count;
            Expr<_T>** v;

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                _T t = (_T)1;
                for (size_t i = 0; i < count; ++i)
                    t *= v[i]->Calc(Map);
                return t;
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Divide : public Expr<_T> {
            Expr<_T>* a;
            Expr<_T>* b;

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                return a->Calc(Map) / b->Calc(Map);
            }
        };

        template <std::integral _T>
        struct DivideRoundUp : public Expr<_T> {
            Expr<_T>* a;
            Expr<_T>* b;

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                _T av = a->Calc(Map);
                _T bv = b->Calc(Map);
                return (av + bv - 1) / bv;
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Mod : public Expr<_T> {
            Expr<_T>* a;
            Expr<_T>* b;

            _T Calc(const std::unordered_map<uint64_t, void*>& Map) {
                _T av = a->Calc(Map);
                _T bv = b->Calc(Map);
                if constexpr (std::floating_point<_T>) {
                    return std::fmod(av, bv);
                }
                else {
                    return av % bc;
                }
            }
        };

        template <size_t _Count = -1>
        struct And : public Expr<bool> {
            Expr<bool>* v[_Count];

            bool Calc(const std::unordered_map<uint64_t, void*>& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t &= v[i]->Calc(Map);
                return t;
            }
        };
        template <>
        struct And<-1> : public Expr<bool> {
            size_t count;
            Expr<bool>** v;

            bool Calc(const std::unordered_map<uint64_t, void*>& Map) {
                bool t = true;
                for (size_t i = 0; i < count; ++i)
                    t &= v[i]->Calc(Map);
                return t;
            }
        };

        template <size_t _Count = -1>
        struct Or : public Expr<bool> {
            Expr<bool>* v[_Count];

            bool Calc(const std::unordered_map<uint64_t, void*>& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t |= v[i]->Calc(Map);
                return t;
            }
        };
        template <>
        struct Or<-1> : public Expr<bool> {
            size_t count;
            Expr<bool>** v;

            bool Calc(const std::unordered_map<uint64_t, void*>& Map) {
                bool t = true;
                for (size_t i = 0; i < count; ++i)
                    t |= v[i]->Calc(Map);
                return t;
            }
        };

        template <size_t _Count = -1>
        struct Xor : public Expr<bool> {
            Expr<bool>* v[_Count];

            bool Calc(const std::unordered_map<uint64_t, void*>& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t ^= v[i]->Calc(Map);
                return t;
            }
        };
        template <>
        struct Xor<-1> : public Expr<bool> {
            size_t count;
            Expr<bool>** v;

            bool Calc(const std::unordered_map<uint64_t, void*>& Map) {
                bool t = true;
                for (size_t i = 0; i < count; ++i)
                    t ^= v[i]->Calc(Map);
                return t;
            }
        };

        struct Not : public Expr<bool> {
            Expr<bool>* v;

            bool Calc(const std::unordered_map<uint64_t, void*>& Map) {
                return !v->Calc(Map);
            }
        };

        namespace details {
            template <typename _TFunc>
            struct FuncParamsTuple;
            template <typename _TOutput, typename... _Ts>
            struct FuncParamsTuple<_TOutput(*)(_Ts...)> {
                using output_t = _TOutput;
                using params_t = std::tuple<_Ts...>;
                using exprptrs_t = std::tuple<Expr<_Ts>*...>;
            };

            template <typename _TTupleI, typename _TTupleO, size_t _Idx = 0>
            void CalcTuple(const _TTupleI& InTuple, _TTupleO& OutTuple, const std::unordered_map<uint64_t, void*>& Map) {
                if constexpr (_Idx >= std::tuple_size_v<_TTupleI>) return;
                using type_t = std::tuple_element_t<_Idx, _TTupleO>;
                new (&std::get<_Idx>(OutTuple)) type_t(std::get<_Idx>(InTuple)->Calc(Map));
                CalcTuple<_TTupleI, _TTupleO, _Idx + 1>(InTuple, OutTuple, Map);
            }
        }

        template <auto _Func>
            requires std::is_function_v<decltype(_Func)>
        struct Func : public Expr<details::FuncParamsTuple<decltype(_Func)>::output_t> {
            using func_t = decltype(_Func);
            using output_t = details::FuncParamsTuple<func_t>::output_t;
            using params_t = details::FuncParamsTuple<func_t>::params_t;
            using exprptrs_t = details::FuncParamsTuple<func_t>::exprptrs_t;

            exprptrs_t params;

            output_t Calc(const std::unordered_map<uint64_t, void*>& Map) {
                std::aligned_storage_t<sizeof(params_t), alignof(params_t)> paramsDat;
                params_t& evaledParams = *(params_t*)&paramsDat;
                
                details::CalcTuple(params, evaledParams, Map);

                return std::apply(_Func, evaledParams);
            }
        };
    }
}