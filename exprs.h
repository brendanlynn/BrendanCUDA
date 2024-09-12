#pragma once

#include <unordered_map>
#include <cmath>
#include <any>

namespace BrendanCUDA {
    namespace Exprs {
        using varmap_t = std::unordered_map<uint64_t, std::any>;

        struct ExprBase {
            virtual std::any CalcToAny(const varmap_t&);
        };
        template <typename _TOutput>
        struct Expr : ExprBase {
            using output_t = _TOutput;

            virtual _TOutput Calc(const varmap_t&);
            std::any CalcToAny(const varmap_t& VarMap) {
                return Calc(VarMap);
            }
        };

        template <auto _Val>
        struct Val : public Expr<decltype(_Val)> {
            auto Calc(const varmap_t&) {
                return _Val;
            }
        };
        template <typename _T>
        struct Val : public Expr<_T> {
            _T val;

            _T Calc(const varmap_t&) {
                return val;
            }
        };
        template <typename _T, uint64_t _Key>
        struct Var : public Expr<_T> {
            constexpr uint64_t key = _Key;

            _T Calc(const varmap_t& Map) {
                return std::any_cast<_T>(Map.at(_Key));
            }
        };
        template <typename _T>
        struct Var : public Expr<_T> {
            uint64_t key;

            _T Calc(const varmap_t& Map) {
                return std::any_cast<_T>(Map.at(key));
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            constexpr size_t size = _Count;

            Expr<_T>* exprs[_Count];

            _T Calc(const varmap_t& Map) {
                _T t = _T{};
                for (size_t i = 0; i < _Count; ++i)
                    t += exprs[i]->Calc(Map);
                return t;
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            std::vector<Expr<_T>*> exprs;

            _T Calc(const varmap_t& Map) {
                _T t = _T{};
                for (size_t i = 0; i < exprs.size(); ++i)
                    t += exprs[i]->Calc(Map);
                return t;
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Subtract : public Expr<_T> {
            Expr<_T>* a;
            Expr<_T>* b;

            _T Calc(const varmap_t& Map) {
                return a->Calc(Map) - b->Calc(Map);
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            constexpr size_t size = _Count;

            Expr<_T>* exprs[_Count];

            _T Calc(const varmap_t& Map) {
                _T t = (_T)1;
                for (size_t i = 0; i < _Count; ++i)
                    t *= exprs[i]->Calc(Map);
                return t;
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            std::vector<Expr<_T>*> exprs;

            _T Calc(const varmap_t& Map) {
                _T t = (_T)1;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t *= exprs[i]->Calc(Map);
                return t;
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Divide : public Expr<_T> {
            Expr<_T>* a;
            Expr<_T>* b;

            _T Calc(const varmap_t& Map) {
                return a->Calc(Map) / b->Calc(Map);
            }
        };

        template <std::integral _T>
        struct DivideRoundUp : public Expr<_T> {
            Expr<_T>* a;
            Expr<_T>* b;

            _T Calc(const varmap_t& Map) {
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

            _T Calc(const varmap_t& Map) {
                _T av = a->Calc(Map);
                _T bv = b->Calc(Map);
                if constexpr (std::floating_point<_T>) {
                    return std::fmod(av, bv);
                }
                else {
                    return av % bv;
                }
            }
        };

        template <size_t _Count = -1>
        struct And : public Expr<bool> {
            constexpr size_t size = _Count;

            Expr<bool>* exprs[_Count];

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t &= exprs[i]->Calc(Map);
                return t;
            }
        };
        template <>
        struct And<-1> : public Expr<bool> {
            std::vector<Expr<bool>*> exprs;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t &= exprs[i]->Calc(Map);
                return t;
            }
        };

        template <size_t _Count = -1>
        struct Or : public Expr<bool> {
            constexpr size_t size = _Count;

            Expr<bool>* exprs[_Count];

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t |= exprs[i]->Calc(Map);
                return t;
            }
        };
        template <>
        struct Or<-1> : public Expr<bool> {
            std::vector<Expr<bool>*> exprs;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t |= exprs[i]->Calc(Map);
                return t;
            }
        };

        template <size_t _Count = -1>
        struct Xor : public Expr<bool> {
            constexpr size_t size = _Count;

            Expr<bool>* v[_Count];

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t ^= v[i]->Calc(Map);
                return t;
            }
        };
        template <>
        struct Xor<-1> : public Expr<bool> {
            std::vector<Expr<bool>*> exprs;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t ^= exprs[i]->Calc(Map);
                return t;
            }
        };

        struct Not : public Expr<bool> {
            Expr<bool>* v;

            bool Calc(const varmap_t& Map) {
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
            void CalcTuple(const _TTupleI& InTuple, _TTupleO& OutTuple, const varmap_t& Map) {
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
            using params_t = details::FuncParamsTuple<func_t>::params_t;
            using exprptrs_t = details::FuncParamsTuple<func_t>::exprptrs_t;

            exprptrs_t params;

            Func<_Func>::output_t Calc(const varmap_t& Map) {
                std::aligned_storage_t<sizeof(params_t), alignof(params_t)> paramsDat;
                params_t& evaledParams = *(params_t*)&paramsDat;
                
                details::CalcTuple(params, evaledParams, Map);

                return std::apply(_Func, evaledParams);
            }
        };
    }
}