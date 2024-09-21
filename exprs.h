#pragma once

#include "arrays.h"
#include <any>
#include <cmath>
#include <memory>
#include <unordered_map>

namespace bcuda {
    namespace exprs {
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

#pragma region ExprImpl
        template <auto _Val>
        struct ValCC : public Expr<decltype(_Val)> {
            ValCC() { }
            ~ValCC() override = default;

            decltype(_Val) Calc(const varmap_t&) override {
                return _Val;
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            ValCC<_Val>* Clone() override {
                return new ValCC<_Val>(*this);
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

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            Val<_T>* Clone() override {
                return new Val<_T>(*this);
            }
        };

        template <typename _T, uint64_t _Key>
        struct VarCC : public Expr<_T> {
            static constexpr uint64_t key = _Key;

            VarCC() { }
            ~VarCC() override = default;

            _T Calc(const varmap_t& Map) override {
                return std::any_cast<_T>(Map.at(_Key));
            }

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            VarCC<_T, _Key>* Clone() override {
                return new VarCC<_T, _Key>(*this);
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

            ArrayV<ExprBase*> GetSubExprs() override {
                return ArrayV<ExprBase*>();
            }

            Var<_T>* Clone() override {
                return new Var<_T>(*this);
            }
        };

        template <typename _T, size_t _Count = (size_t)-1>
            requires std::is_arithmetic_v<_T>
        struct Add : public Expr<_T> {
            static constexpr size_t size = _Count;

            std::unique_ptr<Expr<_T>> exprs[_Count];

            Add(std::initializer_list<Expr<_T>*> Exprs) {
                if (Exprs.size() != _Count)
                    throw std::exception("Exprs.size() must equal _Count");

                std::unique_ptr<Expr<_T>>* p_up = exprs;
                Expr<_T>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Add(const Add<_T, _Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Add(Add<_T, _Count>&&) = default;
            Add<_T, _Count>& operator=(Add<_T, _Count> Other) {
                for (size_t i = 0; i < _Count; ++i) std::swap(exprs[i], Other.exprs[i]);
                return *this;
            }
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
        struct Add<_T, (size_t)-1> : public Expr<_T> {
            std::vector<std::unique_ptr<Expr<_T>>> exprs;

            Add(std::initializer_list<Expr<_T>*> Exprs)
                : exprs(Exprs.size()) {
                std::unique_ptr<Expr<_T>>* p_up = exprs.data();
                Expr<_T>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Add(const Add<_T>& OtherExpr)
                : exprs(OtherExpr.exprs.size()) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Add(Add<_T>&&) = default;
            Add<_T>& operator=(Add<_T> Other) {
                std::swap(exprs, Other.exprs);
                return *this;
            }
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

            Subtract(Expr<_T>* A, Expr<_T>* B)
                : a(A), b(B) { }
            Subtract(std::unique_ptr<Expr<_T>>&& A, std::unique_ptr<Expr<_T>>&& B)
                : a(A), b(B) { }
            Subtract(const Subtract<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            Subtract(Subtract<_T>&&) = default;
            Subtract<_T>& operator=(Subtract<_T> Other) {
                std::swap(a, Other.a);
                std::swap(b, Other.b);
                return *this;
            }
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

        template <typename _T, size_t _Count = (size_t)-1>
            requires std::is_arithmetic_v<_T>
        struct Multiply : public Expr<_T> {
            static constexpr size_t size = _Count;

            std::unique_ptr<Expr<_T>> exprs[_Count];

            Multiply(std::initializer_list<Expr<_T>*> Exprs) {
                if (Exprs.size() != _Count)
                    throw std::exception("Exprs.size() must equal _Count");

                std::unique_ptr<Expr<_T>>* p_up = exprs;
                Expr<_T>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Multiply(const Multiply<_T, _Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Multiply(Multiply<_T, _Count>&&) = default;
            Multiply<_T, _Count>& operator=(Multiply<_T, _Count> Other) {
                for (size_t i = 0; i < _Count; ++i) std::swap(exprs[i], Other.exprs[i]);
                return *this;
            }
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
        struct Multiply<_T, (size_t)-1> : public Expr<_T> {
            std::vector<std::unique_ptr<Expr<_T>>> exprs;

            Multiply(std::initializer_list<Expr<_T>*> Exprs)
                : exprs(Exprs.size()) {
                std::unique_ptr<Expr<_T>>* p_up = exprs.data();
                Expr<_T>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Multiply(const Multiply<_T>& OtherExpr)
                : exprs(OtherExpr.exprs.size()) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<_T>>(OtherExpr.exprs[i]->Clone());
            }
            Multiply(Multiply<_T>&&) = default;
            Multiply<_T>& operator=(Multiply<_T> Other) {
                std::swap(exprs, Other.exprs);
                return *this;
            }
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

            Divide(Expr<_T>* A, Expr<_T>* B)
                : a(A), b(B) { }
            Divide(std::unique_ptr<Expr<_T>>&& A, std::unique_ptr<Expr<_T>>&& B)
                : a(A), b(B) { }
            Divide(const Divide<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            Divide(Divide<_T>&&) = default;
            Divide<_T>& operator=(Divide<_T> Other) {
                std::swap(a, Other.a);
                std::swap(b, Other.b);
                return *this;
            }
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

            DivideRoundUp(Expr<_T>* A, Expr<_T>* B)
                : a(A), b(B) { }
            DivideRoundUp(std::unique_ptr<Expr<_T>>&& A, std::unique_ptr<Expr<_T>>&& B)
                : a(A), b(B) { }
            DivideRoundUp(const DivideRoundUp<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            DivideRoundUp(DivideRoundUp<_T>&&) = default;
            DivideRoundUp<_T>& operator=(DivideRoundUp<_T> Other) {
                std::swap(a, Other.a);
                std::swap(b, Other.b);
                return *this;
            }
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

            Mod(Expr<_T>* A, Expr<_T>* B)
                : a(A), b(B) { }
            Mod(std::unique_ptr<Expr<_T>>&& A, std::unique_ptr<Expr<_T>>&& B)
                : a(A), b(B) { }
            Mod(const Mod<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            Mod(Mod<_T>&&) = default;
            Mod<_T>& operator=(Mod<_T> Other) {
                std::swap(a, Other.a);
                std::swap(b, Other.b);
                return *this;
            }
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

            ModBlock(Expr<_T>* A, Expr<_T>* B)
                : a(A), b(B) { }
            ModBlock(std::unique_ptr<Expr<_T>>&& A, std::unique_ptr<Expr<_T>>&& B)
                : a(A), b(B) { }
            ModBlock(const ModBlock<_T>& OtherExpr)
                : a(OtherExpr.a->Clone()), b(OtherExpr.b->Clone()) { }
            ModBlock(ModBlock<_T>&&) = default;
            ModBlock<_T>& operator=(ModBlock<_T> Other) {
                std::swap(a, Other.a);
                std::swap(b, Other.b);
                return *this;
            }
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

        template <size_t _Count = (size_t)-1>
        struct And : public Expr<bool> {
            static constexpr size_t size = _Count;

            std::unique_ptr<Expr<bool>> exprs[_Count];

            And(std::initializer_list<Expr<bool>*> Exprs) {
                if (Exprs.size() != _Count)
                    throw std::exception("Exprs.size() must equal _Count");

                std::unique_ptr<Expr<bool>>* p_up = exprs;
                Expr<bool>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            And(const And<_Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            And(And<_Count>&&) = default;
            And<_Count>& operator=(And<_Count> Other) {
                for (size_t i = 0; i < _Count; ++i) std::swap(exprs[i], Other.exprs[i]);
                return *this;
            }
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
        struct And<(size_t)-1> : public Expr<bool> {
            std::vector<std::unique_ptr<Expr<bool>>> exprs;

            And(std::initializer_list<Expr<bool>*> Exprs)
                : exprs(Exprs.size()) {
                std::unique_ptr<Expr<bool>>* p_up = exprs.data();
                Expr<bool>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            And(const And& OtherExpr)
                : exprs(OtherExpr.exprs.size()) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            And(And&&) = default;
            And& operator=(And Other) {
                std::swap(exprs, Other.exprs);
                return *this;
            }
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

            And* Clone() override {
                return new And(*this);
            }
        };

        template <size_t _Count = (size_t)-1>
        struct Or : public Expr<bool> {
            static constexpr size_t size = _Count;

            std::unique_ptr<Expr<bool>> exprs[_Count];

            Or(std::initializer_list<Expr<bool>*> Exprs) {
                if (Exprs.size() != _Count)
                    throw std::exception("Exprs.size() must equal _Count");

                std::unique_ptr<Expr<bool>>* p_up = exprs;
                Expr<bool>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Or(const Or<_Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            Or(Or<_Count>&&) = default;
            Or<_Count>& operator=(Or<_Count> Other) {
                for (size_t i = 0; i < _Count; ++i) std::swap(exprs[i], Other.exprs[i]);
                return *this;
            }
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
        struct Or<(size_t)-1> : public Expr<bool> {
            std::vector<std::unique_ptr<Expr<bool>>> exprs;

            Or(std::initializer_list<Expr<bool>*> Exprs)
                : exprs(Exprs.size()) {
                std::unique_ptr<Expr<bool>>* p_up = exprs.data();
                Expr<bool>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Or(const Or& OtherExpr)
                : exprs(OtherExpr.exprs.size()) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            Or(Or&&) = default;
            Or& operator=(Or Other) {
                std::swap(exprs, Other.exprs);
                return *this;
            }
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

            Or<(size_t)-1>* Clone() override {
                return new Or<(size_t)-1>(*this);
            }
        };

        template <size_t _Count = (size_t)-1>
        struct Xor : public Expr<bool> {
            static constexpr size_t size = _Count;

            std::unique_ptr<Expr<bool>> exprs[_Count];

            Xor(std::initializer_list<Expr<bool>*> Exprs) {
                if (Exprs.size() != _Count)
                    throw std::exception("Exprs.size() must equal _Count");

                std::unique_ptr<Expr<bool>>* p_up = exprs;
                Expr<bool>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Xor(const Xor<_Count>& OtherExpr) {
                for (size_t i = 0; i < _Count; ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            Xor(Xor<_Count>&&) = default;
            Xor<_Count>& operator=(Xor<_Count> Other) {
                for (size_t i = 0; i < _Count; ++i) std::swap(exprs[i], Other.exprs[i]);
                return *this;
            }
            ~Xor() override = default;

            bool Calc(const varmap_t& Map) override {
                bool t = true;
                for (size_t i = 0; i < _Count; ++i)
                    t ^= exprs[i]->Calc(Map);
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
        struct Xor<(size_t)-1> : public Expr<bool> {
            std::vector<std::unique_ptr<Expr<bool>>> exprs;

            Xor(std::initializer_list<Expr<bool>*> Exprs)
                : exprs(Exprs.size()) {
                std::unique_ptr<Expr<bool>>* p_up = exprs.data();
                Expr<bool>* const* p_op = Exprs.begin();
                for (; p_op < Exprs.end(); ++p_up, ++p_op)
                    p_up->reset(*p_op);
            }
            Xor(const Xor& OtherExpr)
                : exprs(OtherExpr.exprs.size()) {
                for (size_t i = 0; i < OtherExpr.exprs.size(); ++i)
                    exprs[i] = std::unique_ptr<Expr<bool>>(OtherExpr.exprs[i]->Clone());
            }
            Xor(Xor&&) = default;
            Xor& operator=(Xor Other) {
                std::swap(exprs, Other.exprs);
                return *this;
            }
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

            Xor* Clone() override {
                return new Xor(*this);
            }
        };

        struct Not : public Expr<bool> {
            std::unique_ptr<Expr<bool>> v;

            Not(Expr<bool>* V)
                : v(V) { }
            Not(std::unique_ptr<Expr<bool>>&& V)
                : v(std::move(V)) { }
            Not(const Not& OtherExpr)
                : v(OtherExpr.v->Clone()) { }
            Not(Not&&) = default;
            Not& operator=(Not Other) {
                std::swap(v, Other.v);
                return *this;
            }
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
#pragma endregion

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
        struct Func : public Expr<typename details::FuncParamsTuple<decltype(_Func)>::output_t> {
            using func_t = decltype(_Func);
            using params_t = details::FuncParamsTuple<func_t>::params_t;
            using exprptrs_t = details::FuncParamsTuple<func_t>::exprptrs_t;
            static constexpr size_t paramCount = std::tuple_size_v<params_t>;

            exprptrs_t params;

            Func(exprptrs_t&& Exprs)
                : params(std::move(Exprs)) { }
            Func(const Func<_Func>& OtherExpr) {
                details::CloneTuple(OtherExpr.params, params);
            }
            Func(Func<_Func>&&) = default;
            Func<_Func>& operator=(Func<_Func> Other) {
                std::swap(params, Other.params);
                return *this;
            }
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

            Func<_Func>* Clone() override {
                return new Func<_Func>(*this);
            }
        };
    }
    template <typename _T>
    struct Expression {
        std::unique_ptr<exprs::Expr<_T>> ptr;

        Expression(std::nullptr_t = nullptr)
            : ptr(nullptr) { }
        Expression(std::unique_ptr<exprs::Expr<_T>>&& Ptr)
            : ptr(Ptr) { }
        Expression(const Expression<_T>& OtherExpr)
            : ptr(OtherExpr.ptr->Clone()) { }
        Expression(Expression<_T>&&) = default;

        Expression<_T>& operator=(Expression<_T> Other) {
            std::swap(ptr, Other.ptr);
            return *this;
        }

        _T Calculate(const exprs::varmap_t& Map) {
            return ptr->Calc(Map);
        }
    };
}