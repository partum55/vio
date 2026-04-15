#pragma once

#include <atomic>
#include <future>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <stdexcept>
#include "thread_safe_queue.hpp"
#include "function_wrapper.hpp"
#include "abc_thread_pool.hpp"

//based on C++ Concurrency in Action, Ch. 9.1 by Williams
class ThreadPool final : public ABCThreadPool {
public:
    explicit ThreadPool(int thread_count)
        : stopping_(false)
    {
        if (thread_count <= 0) {
            thread_count = 1;
        }
        workers_.reserve(static_cast<size_t>(thread_count));

        for (int i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                worker_loop();
            });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    ~ThreadPool() noexcept {
        shutdown();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void shutdown() noexcept {
        bool expected = false;
        if (stopping_.compare_exchange_strong(expected, true)) {
            work_queue_.close();
        }
    }

    template <class F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
        using result_type = std::invoke_result_t<std::decay_t<F>>;

        if (stopping_.load()) {
            throw std::runtime_error("ThreadPool is stopping; submit() is not allowed.");
        }

        std::packaged_task<result_type()> task(std::forward<F>(f));
        std::future<result_type> res = task.get_future();

        if (!work_queue_.emplace(FunctionWrapper([t = std::move(task)]() mutable { t(); }))) {
            throw std::runtime_error("ThreadPool queue is closed");
        }
        return res;
    }

    std::future<void> submit_task(std::function<void()> task) override {
        return submit(std::move(task));
    }

    bool run_pending_task() override {
        auto task = work_queue_.try_deque();
        if (!task) return false;
        (*task)();
        return true;
    }

private:
    void worker_loop() {
        while (true) {
            auto task = work_queue_.deque();
            if (!task) break;
            (*task)();
        }
    }

    ThreadSafeQueue<FunctionWrapper> work_queue_;
    std::vector<std::thread> workers_;
    std::atomic_bool stopping_;
};