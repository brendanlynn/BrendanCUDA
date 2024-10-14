#pragma once

#include <type_traits>

namespace bcuda {
    namespace packs {
        namespace details {
            template <uintmax_t _Idx, typename... _Ts>
            struct typeAt;
            template <typename _T, typename... _Ts>
            struct typeAt<0, _T, _Ts...> {
                using type_t = _T;
            };
            template <uintmax_t _Idx, typename _T, typename... _Ts>
            struct typeAt<_Idx, _T, _Ts...>
                : typeAt<_Idx - 1, _Ts...> { };
        }

        template <typename... _Ts>
        struct Pack final {
            template <uintmax_t _Idx>
            using param_t = details::typeAt<_Idx, _Ts...>;
        };

        namespace details {
            template <typename _T>
            struct isPack : std::false_type { };
            template <typename... _Ts>
            struct isPack<Pack<_Ts...>> : std::true_type { };
        }

        template <typename _T>
        concept IsPack = details::isPack<_T>::value;

        template <typename _T>
        struct PassingParam final {
            using inside_t = _T;
        };
        template <typename _T>
        using passParam_t = PassingParam<_T>;

        namespace details {
            template <typename _T, uintmax_t _LayerCount>
            struct RecursivePassingParam {
                using type_t = PassingParam<typename RecursivePassingParam<_T, _LayerCount - 1>::type_t>;
            };
            template <typename _T>
            struct RecursivePassingParam<_T, 0> {
                using type_t = _T;
            };
        }

        template <IsPack _T, uintmax_t _LayerCount>
        using recursivePassingParams_t = typename details::RecursivePassingParam<_T, _LayerCount>::type_t;

        namespace details {
            template <typename _T>
            struct bringOut {
                using type_t = _T;
            };
            template <typename _T>
            struct bringOut<PassingParam<_T>> {
                using type_t = _T;
            };

            template <typename _T>
            struct bringOutBatch;
            template <typename... _Ts>
            struct bringOutBatch<Pack<_Ts...>> {
                using type_t = Pack<bringOut<_Ts>, ...>;
            };
        }

        template <typename _T>
        using bringOutInside_t = typename details::bringOut<_T>::type_t;
        template <typename _T>
        using bringOutInsideBatch_t = typename details::bringOutBatch<_T>::type_t;

        namespace details {
            template <typename _T>
            struct isPackSatisfyingAll : std::false_type { };
            template <template <typename> typename _TPredicate, typename... _Ts>
            struct isPackSatisfyingAll : std::bool_constant<_TPredicate<_Ts>::value && ...> { };
        }

        template <typename _T, template <typename> typename _TPredicate>
        concept IsPackAndElementsSatisfy = details::isPackSatisfyingAll<_T>::value;

        namespace details {
            template <typename... _Ts>
            struct addPacks;

            template <>
            struct addPacks<> {
                using type_t = Pack<>;
            };

            template <typename _T>
            struct addPacks<_T> {
                using type_t = Pack<bringOutInside_t<_T>>;
            };

            template <typename... _Ts>
            struct addPacks<Pack<_Ts...>> {
                using type_t = Pack<_Ts...>;
            };

            template <typename _T1, typename _T2>
            struct addPacks<_T1, _T2> {
                using type_t = Pack<bringOutInside_t<_T1>, bringOutInside_t<_T2>>;
            };

            template <typename _T, typename... _Ts>
            struct addPacks<_T, Pack<_Ts...>> {
                using type_t = Pack<bringOutInside_t<_T>, _Ts...>;
            };

            template <typename... _Ts, typename _T>
            struct addPacks<Pack<_Ts...>, _T> {
                using type_t = Pack<_Ts..., bringOutInside_t<_T>>;
            };

            template <typename... _Ts1, typename... _Ts2>
            struct addPacks<Pack<_Ts1...>, Pack<_Ts2...>> {
                using type_t = Pack<_Ts1..., _Ts2...>;
            };

            template <typename _T1, typename _T2, typename... _Ts>
            struct addPacks<_T1, _T2, _Ts...> {
                using type_t = typename addPacks<addPacks<_T1, _T2>::type_t, _Ts...>::type_t;
            };
        }

        template <typename... _Ts>
        using combine_t = typename details::addPacks<_Ts...>::type_t;

        namespace details {
            template <template <typename...> typename _T, typename _TPack>
            struct applyPack;
            template <template <typename...> typename _T, typename... _Ts>
            struct applyPack<_T, Pack<_Ts...>> {
                using type_t = _T<_Ts...>;
            };
        }

        template <template <typename...> typename _T, IsPack _TPack>
        using appliedPack_t = typename details::applyPack<_T, _Pack>::type_t;
    }
}