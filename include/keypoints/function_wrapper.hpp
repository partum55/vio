#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace vio {

class FunctionWrapper final {
    struct ImplBase {
        virtual void call() = 0;
        virtual ~ImplBase() = default;
    };

    template <class F>
    struct ImplType final : ImplBase {
        F f;

        explicit ImplType(F&& f_) : f(std::move(f_)) {}

        void call() override {
            f();
        }
    };

    std::unique_ptr<ImplBase> impl_;

public:
    FunctionWrapper() = default;

    template <class F>
    explicit FunctionWrapper(F&& f)
        : impl_(std::make_unique<ImplType<std::decay_t<F>>>(std::forward<F>(f))) {}

    void operator()() {
        if (impl_) {
            impl_->call();
        }
    }

    FunctionWrapper(FunctionWrapper&& other) noexcept = default;
    FunctionWrapper& operator=(FunctionWrapper&& other) noexcept = default;

    FunctionWrapper(const FunctionWrapper&) = delete;
    FunctionWrapper& operator=(const FunctionWrapper&) = delete;
};

} // namespace vio
