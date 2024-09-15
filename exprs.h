#pragma once

#include "arrays.h"
#include <any>
#include <cmath>
#include <memory>
#include <unordered_map>

namespace BrendanCUDA {
    namespace Exprs {
        using varmap_t = std::unordered_map<uint64_t, std::any>;

        struct ExprBase {
            virtual ~ExprBase() = default;

            virtual std::any CalcToAny(const varmap_t&) = 0;

            virtual ArrayV<ExprBase*> GetSubExprs() = 0;

            virtual ExprBase* Clone() = 0;
        };
        template <typename _TOutput>
        struct Expr : ExprBase {
            using output_t = _TOutput;

            virtual _TOutput Calc(const varmap_t&) = 0;

            std::any CalcToAny(const varmap_t& VarMap) override {
                return Calc(VarMap);
            }

            virtual Expr<_TOutput>* Clone() override = 0;
        };

        template <auto _Val>
        struct Val : public Expr<decltype(_Val)> {
            Val() { }
            ~Val() override = default;

            auto Calc(const varmap_t&) override {
                return _Val;
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            Val<_Val>* Clone() override {
                return new Val<_Val>(*this);
            }
        };
        template <typename _T>
        struct Val : public Expr<_T> {
            _T val;

            Val(_T Val)
                : val(Val) { }
            ~Val() override = default;

            _T Calc(const varmap_t&) override {
                return val;
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            Val<_T>* Clone() override {
                return new Val<_T>(*this);
            }
        };

        template <typename _T, uint64_t _Key>
        struct Var : public Expr<_T> {
            constexpr uint64_t key = _Key;

            Var() { }
            ~Var() override = default;

            _T Calc(const varmap_t& Map) override {
                return std::any_cast<_T>(Map.at(_Key));
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            Var<_T, _Key>* Clone() override {
                return new Var<_T, _Key>(*this);
            }
        };
        template <typename _T>
        struct Var : public Expr<_T> {
            uint64_t key;

            Var(uint64_t Key)
                : key(Key) { }
            ~Var() override = default;

            _T Calc(const varmap_t& Map) override {
                return std::any_cast<_T>(Map.at(key));
            }

            auto GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            Var<_T>* Clone() override {
                return new Var<_T>(*this);
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            constexpr size_t size = _Count;

            std::unique_ptr<Expr<_T>> exprs[_Count];

            Add(const Add<_T, _Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Add(Add<_T, _Count>&&) = default;
            ~Add() override = default;

            _T Calc(const varmap_t& Map) override {
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

            Add<_T, _Count>* Clone() override {
                return Add<_T, _Count>(*this);
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            std::vector<std::unique_ptr<Expr<_T>>> exprs;

            Add(const Add<_T>& OtherExpr) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Add(Add<_T>&&) = default;
            ~Add() override = default;

            _T Calc(const varmap_t& Map) override {
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

            Add<_T>* Clone() override {
                return new Add<_T>(*this);
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Subtract : public Expr<_T> {
            std::unique_ptr<Expr<_T>> a;
            std::unique_ptr<Expr<_T>> b;

            Subtract(const Subtract<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            Subtract(Subtract<_T>&&) = default;
            ~Subtract() override = default;

            _T Calc(const varmap_t& Map) override {
                return a->Calc(Map) - b->Calc(Map);
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }

            Subtract<_T>* Clone() override {
                return new Subtract<_T>(*this);
            }
        };

        template <typename _T, size_t _Count>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            constexpr size_t size = _Count;

            std::unique_ptr<Expr<_T>> exprs[_Count];

            Multiply(const Multiply<_T, _Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Multiply(Multiply<_T, _Count>&&) = default;
            ~Multiply() override = default;

            _T Calc(const varmap_t& Map) override {
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

            Multiply<_T, _Count>* Clone() override {
                return new Multiply<_T, _Count>(*this);
            }
        };
        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            std::vector<std::unique_ptr<Expr<_T>>> exprs;

            Multiply(const Multiply<_T>& OtherExpr) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Multiply(Multiply<_T>&&) = default;
            ~Multiply() override = default;

            _T Calc(const varmap_t& Map) override {
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

            Multiply<_T>* Clone() override {
                return new Multiply<_T>(*this);
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Divide : public Expr<_T> {
            std::unique_ptr<Expr<_T>> a;
            std::unique_ptr<Expr<_T>> b;

            Divide(const Divide<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            Divide(Divide<_T>&&) = default;
            ~Divide() override = default;

            _T Calc(const varmap_t& Map) {
                return a->Calc(Map) / b->Calc(Map);
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }

            Divide<_T>* Clone() override {
                return new Divide<_T>(*this);
            }
        };

        template <std::integral _T>
        struct DivideRoundUp : public Expr<_T> {
            std::unique_ptr<Expr<_T>> a;
            std::unique_ptr<Expr<_T>> b;

            DivideRoundUp(const DivideRoundUp<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            DivideRoundUp(DivideRoundUp<_T>&&) = default;
            ~DivideRoundUp() override = default;

            _T Calc(const varmap_t& Map) override {
                _T av = a->Calc(Map);
                _T bv = b->Calc(Map);
                return (av + bv - 1) / bv;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ a.Get(), b.Get() });
            }

            DivideRoundUp<_T>* Clone() override {
                return new DivideRoundUp<_T>(*this);
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct Mod : public Expr<_T> {
            std::unique_ptr<Expr<_T>> a;
            std::unique_ptr<Expr<_T>> b;

            Mod(const Mod<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            Mod(Mod<_T>&&) = default;
            ~Mod() override = default;

            _T Calc(const varmap_t& Map) override {
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

            Mod<_T>* Clone() override {
                return new Mod<_T>(*this);
            }
        };

        template <typename _T>
            requires std::is_arithmetic_v<_T>
        struct ModBlock : public Expr<_T> {
            std::unique_ptr<Expr<_T>> a;
            std::unique_ptr<Expr<_T>> b;

            ModBlock(const ModBlock<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            ModBlock(ModBlock<_T>&&) = default;
            ~ModBlock() override = default;

            _T Calc(const varmap_t& Map) override {
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

            ModBlock<_T>* Clone() override {
                return new ModBlock<_T>(*this);
            }
        };

        template <size_t _Count = -1>
        struct And : public Expr<bool> {
            constexpr size_t size = _Count;

            std::unique_ptr<Expr<bool>> exprs[_Count];

            And(const And<_Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            And(And<_Count>&&) = default;
            ~And() override = default;

            bool Calc(const varmap_t& Map) override {
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

            And<_Count>* Clone() override {
                return new And<_Count>(*this);
            }
        };
        template <>
        struct And<-1> : public Expr<bool> {
            std::vector<std::unique_ptr<Expr<bool>>> exprs;

            And(const And<-1>& OtherExpr) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            And(And<-1>&&) = default;
            ~And() override = default;

            bool Calc(const varmap_t& Map) override {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t &= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].get();
                return arr;
            }

            And<-1>* Clone() override {
                return new And<-1>(*this);
            }
        };

        template <size_t _Count = -1>
        struct Or : public Expr<bool> {
            constexpr size_t size = _Count;

            std::unique_ptr<Expr<bool>> exprs[_Count];

            Or(const Or<_Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Or(Or<_Count>&&) = default;
            ~Or() override = default;

            bool Calc(const varmap_t& Map) override {
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

            Or<_Count>* Clone() override {
                return new Or<_Count>(*this);
            }
        };
        template <>
        struct Or<-1> : public Expr<bool> {
            std::vector<std::unique_ptr<Expr<bool>>> exprs;

            Or(const Or<-1>& OtherExpr) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            Or(Or<-1>&&) = default;
            ~Or() override = default;

            bool Calc(const varmap_t& Map) override {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t |= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].get();
                return arr;
            }

            Or<-1>* Clone() override {
                return new Or<-1>(*this);
            }
        };

        template <size_t _Count = -1>
        struct Xor : public Expr<bool> {
            constexpr size_t size = _Count;

            std::unique_ptr<Expr<bool>> v[_Count];

            Xor(const Xor<_Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Xor(Xor<_Count>&&) = default;
            ~Xor() override = default;

            bool Calc(const varmap_t& Map) override {
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

            Xor<_Count>* Clone() override {
                return new Xor<_Count>(*this);
            }
        };
        template <>
        struct Xor<-1> : public Expr<bool> {
            std::vector<std::unique_ptr<Expr<bool>>> exprs;

            Xor(const Xor<-1>& OtherExpr) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            Xor(Xor<-1>&&) = default;
            ~Xor() override = default;

            bool Calc(const varmap_t& Map) override {
                bool t = true;
                for (size_t i = 0; i < exprs.size(); ++i)
                    t ^= exprs[i]->Calc(Map);
                return t;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                ArrayV<ExprBase*> arr(exprs.size());
                for (size_t i = 0; i < arr.Size(); ++i) arr[i] = exprs[i].get();
                return arr;
            }

            Xor<-1>* Clone() override {
                return new Xor<-1>(*this);
            }
        };

        struct Not : public Expr<bool> {
            std::unique_ptr<Expr<bool>> v;

            Not(const Not& OtherExpr)
                : v(OtherExpr.v->Clone()) { }
            Not(Not&&) = default;
            ~Not() override = default;

            bool Calc(const varmap_t& Map) override {
                return !v->Calc(Map);
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>({ v.get() });
            }

            Not* Clone() override {
                return new Not(*this);
            }
        };

        namespace details {
            template <typename _TFunc>
            struct FuncParamsTuple;
            template <typename _TOutput, typename... _Ts>
            struct FuncParamsTuple<_TOutput(*)(_Ts...)> {
                using output_t = _TOutput;
                using params_t = std::tuple<_Ts...>;
                using exprptrs_t = std::tuple<std::unique_ptr<Expr<_Ts>>...>;
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

            template <typename _TTuple, size_t _Idx = 0>
            __forceinline void CloneTuple(const _TTuple& TupleOld, _TTuple& TupleNew) {
                using type_t = std::tuple_element_t<_Idx, _TTuple>;
                std::get<_Idx>(TupleNew) = type_t(std::get<_Idx>(TupleOld)->Clone());
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

            Func(const Func<_Func>& OtherExpr) {
                details::CloneTuple(OtherExpr.params, params);
            }
            Func(Func<_Func>&&) = default;
            ~Func() override = default;

            Func<_Func>::output_t Calc(const varmap_t& Map) override {
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

            Func<_Func>* Clone() {
                return new Func<_Func>(*this);
            }
        };
    }
}