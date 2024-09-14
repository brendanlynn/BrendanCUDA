#pragma once

#include "arrays.h"
#include "copyptr.h"
#include <any>
#include <cmath>
#include <unordered_map>

namespace BrendanCUDA {
    namespace Exprs {
        using varmap_t = std::unordered_map<uint64_t, std::any>;

        struct ExprBase {
            virtual ~ExprBase() = default;

            virtual std::any CalcToAny(const varmap_t&) = 0;

            virtual ArrayV<ExprBase*> GetSubExprs() = 0;
        };
        template <typename _TOutput>
        struct Expr : ExprBase {
            using output_t = _TOutput;

            virtual _TOutput Calc(const varmap_t&) = 0;

            std::any CalcToAny(const varmap_t& VarMap) override {
                return Calc(VarMap);
            }
        };

        template <auto _Val>
        struct Val : public Expr<decltype(_Val)> {
            ~Val() override = default;

            auto Calc(const varmap_t&) override {
                return _Val;
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }
        };
        template <typename _T>
        struct Val : public Expr<_T> {
            _T val;

            ~Val() override = default;

            _T Calc(const varmap_t&) {
                return val;
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }
        };

        template <typename _T, uint64_t _Key>
        struct Var : public Expr<_T> {
            constexpr uint64_t key = _Key;

            ~Var() override = default;

            _T Calc(const varmap_t& Map) {
                return std::any_cast<_T>(Map.at(_Key));
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }
        };
        template <typename _T>
        struct Var : public Expr<_T> {
            uint64_t key;

            ~Var() override = default;

            _T Calc(const varmap_t& Map) {
                return std::any_cast<_T>(Map.at(key));
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            constexpr size_t size = _Count;

            CopyPtr<Expr<_T>> exprs[_Count];

            ~Add() override = default;

            _T Calc(const varmap_t& Map) {
                _T t = _T{};
                for (size_t i = 0; i < _Count; ++i)
                    t += exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(_Count);
                for (size_t i = 0; i < _Count; ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            std::vector<CopyPtr<Expr<_T>>> exprs;

            ~Add() override = default;

            _T Calc(const varmap_t& Map) {
                _T t = _T{};
                for (size_t i = 0; i < exprs.size(); ++i)
                    t += exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Subtract : public Expr<_T> {
            CopyPtr<Expr<_T>> a;
            CopyPtr<Expr<_T>> b;

            ~Subtract() override = default;

            _T Calc(const varmap_t& Map) {
                return a->Calc(Map) - b->Calc(Map);
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            constexpr size_t size = _Count;

            CopyPtr<Expr<_T>> exprs[_Count];

            ~Multiply() override = default;

            _T Calc(const varmap_t& Map) {
                _T t = (_T)1;
                for (size_t i = 0; i < _Count; ++i)
                    t *= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(_Count);
                for (size_t i = 0; i < _Count; ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            std::vector<CopyPtr<Expr<_T>>> exprs;

            ~Multiply() override = default;

            _T Calc(const varmap_t& Map) {
                _T t = (_T)1;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t *= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Divide : public Expr<_T> {
            CopyPtr<Expr<_T>> a;
            CopyPtr<Expr<_T>> b;

            ~Divide() override = default;

            _T Calc(const varmap_t& Map) {
                return a->Calc(Map) / b->Calc(Map);
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }
        };

        template <std::integral _T>
        struct DivideRoundUp : public Expr<_T> {
            CopyPtr<Expr<_T>> a;
            CopyPtr<Expr<_T>> b;

            ~DivideRoundUp() override = default;

            _T Calc(const varmap_t& Map) {
                _T av = a->Calc(Map);
                _T bv = b->Calc(Map);
                return (av + bv - 1) / bv;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Mod : public Expr<_T> {
            CopyPtr<Expr<_T>> a;
            CopyPtr<Expr<_T>> b;

            ~Mod() override = default;

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

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct ModBlock : public Expr<_T> {
            CopyPtr<Expr<_T>> a;
            CopyPtr<Expr<_T>> b;

            ~ModBlock() override = default;

            _T Calc(const varmap_t& Map) {
                _T av = a->Calc(Map);
                _T bv = b->Calc(Map);

                _T v;
                if constexpr (std::floating_point<_T>) v = std::fmod(av, bv);
                else v = av % bv;

                return v < (_T)0 ? v + bv : v;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }
        };

        template <size_t _Count = -1>
        struct And : public Expr<bool> {
            constexpr size_t size = _Count;

            CopyPtr<Expr<bool>> exprs[_Count];

            ~And() override = default;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t &= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(_Count);
                for (size_t i = 0; i < _Count; ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };
        template <>
        struct And<-1> : public Expr<bool> {
            std::vector<CopyPtr<Expr<bool>>> exprs;

            ~And() override = default;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t &= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };

        template <size_t _Count = -1>
        struct Or : public Expr<bool> {
            constexpr size_t size = _Count;

            CopyPtr<Expr<bool>> exprs[_Count];

            ~Or() override = default;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t |= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(_Count);
                for (size_t i = 0; i < _Count; ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };
        template <>
        struct Or<-1> : public Expr<bool> {
            std::vector<CopyPtr<Expr<bool>>> exprs;

            ~Or() override = default;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t |= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };

        template <size_t _Count = -1>
        struct Xor : public Expr<bool> {
            constexpr size_t size = _Count;

            CopyPtr<Expr<bool>> v[_Count];

            ~Xor() override = default;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t ^= v[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(_Count);
                for (size_t i = 0; i < _Count; ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };
        template <>
        struct Xor<-1> : public Expr<bool> {
            std::vector<CopyPtr<Expr<bool>>> exprs;

            ~Xor() override = default;

            bool Calc(const varmap_t& Map) {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t ^= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].Get();
                return arr;
            }
        };

        struct Not : public Expr<bool> {
            CopyPtr<Expr<bool>> v;

            ~Not() override = default;

            bool Calc(const varmap_t& Map) {
                return !v->Calc(Map);
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ v.Get() });
            }
        };

        namespace details {
            template <typename _TFunc>
            struct FuncParamsTuple;
            template <typename _TOutput, typename... _Ts>
            struct FuncParamsTuple<_TOutput(*)(_Ts...)> {
                using output_t = _TOutput;
                using params_t = std::tuple<_Ts...>;
                using exprptrs_t = std::tuple<CopyPtr<Expr<_Ts>>...>;
            };

            template <typename _TTupleI, typename _TTupleO, size_t _Idx = 0>
            __forceinline void CalcTuple(const _TTupleI& InTuple, _TTupleO& OutTuple, const varmap_t& Map) {
                if constexpr (_Idx >= std::tuple_size_v<_TTupleI>) return;
                using type_t = std::tuple_element_t<_Idx, _TTupleO>;
                new (&std::get<_Idx>(OutTuple)) type_t(std::get<_Idx>(InTuple)->Calc(Map));
                CalcTuple<_TTupleI, _TTupleO, _Idx + 1>(InTuple, OutTuple, Map);
            }

            template <typename _TTuple, size_t _Idx = 0>
            __forceinline void ConvertTupleToArray(const _TTuple& Tuple, ArrayV<ExprBase*>& Array) {
                if constexpr (_Idx >= std::tuple_size_v<_TTuple>) return;
                Array[_Idx] = std::get<_Idx>(Tuple);
                ConvertTupleToArray<_TTuple, _Idx + 1>(Tuple, Array);
            }
        }

        template <auto _Func>
            requires std::is_function_v<decltype(_Func)>
        struct Func : public Expr<details::FuncParamsTuple<decltype(_Func)>::output_t> {
            using func_t = decltype(_Func);
            using params_t = details::FuncParamsTuple<func_t>::params_t;
            using exprptrs_t = details::FuncParamsTuple<func_t>::exprptrs_t;
            constexpr paramCount = std::tuple_size_v<params_t>;

            exprptrs_t params;

            ~Func() override = default;

            Func<_Func>::output_t Calc(const varmap_t& Map) {
                std::aligned_storage_t<sizeof(params_t), alignof(params_t)> paramsDat;
                params_t& evaledParams = *(params_t*)&paramsDat;
                
                details::CalcTuple(params, evaledParams, Map);

                return std::apply(_Func, evaledParams);
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(paramCount);
                details::ConvertTupleToArray(params, arr);
                return arr;
            }
        };
    }
}