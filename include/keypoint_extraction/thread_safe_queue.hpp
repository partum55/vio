#pragma once

#include <oneapi/tbb/concurrent_queue.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

namespace vio {

template <class T>
class ThreadSafeQueue final {
public:
    ThreadSafeQueue() = default;

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue(ThreadSafeQueue&&) = delete;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;

    ~ThreadSafeQueue() noexcept {
        close();
    }

    //copy
    bool enque(const T& value) {
        if (closed_.load(std::memory_order_acquire)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(state_m_);
            if (closed_.load(std::memory_order_relaxed)) {
                return false;
            }
            q_.push(value);
            size_.fetch_add(1, std::memory_order_release);
        }

        cv_.notify_one();
        return true;
    }

    //move
    bool enque(T&& value) {
        if (closed_.load(std::memory_order_acquire)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(state_m_);
            if (closed_.load(std::memory_order_relaxed)) {
                return false;
            }
            q_.push(std::move(value));
            size_.fetch_add(1, std::memory_order_release);
        }

        cv_.notify_one();
        return true;
    }

    template <class... Args>
    bool emplace(Args&&... args) {
        return enque(T(std::forward<Args>(args)...));
    }

    //blocking
    std::optional<T> deque() {
        while (true) {
            T value;
            if (q_.try_pop(value)) {
                size_.fetch_sub(1, std::memory_order_acq_rel);
                cv_.notify_all();
                return value;
            }

            std::unique_lock<std::mutex> ul(wait_m_);
            cv_.wait(ul, [this] {
                return closed_.load(std::memory_order_acquire) ||
                       size_.load(std::memory_order_acquire) > 0;
            });

            if (closed_.load(std::memory_order_acquire) &&
                size_.load(std::memory_order_acquire) == 0) {
                return std::nullopt;
            }
        }
    }

    //doesn`t wait
    std::optional<T> try_deque() {
        T value;
        if (!q_.try_pop(value)) {
            return std::nullopt;
        }

        size_.fetch_sub(1, std::memory_order_acq_rel);
        cv_.notify_all();
        return value;
    }

    bool empty() const {
        return size_.load(std::memory_order_acquire) == 0;
    }

    std::size_t size() const {
        return size_.load(std::memory_order_acquire);
    }

    void close() noexcept {
        const bool was_open = !closed_.exchange(true, std::memory_order_acq_rel);
        if (was_open) {
            cv_.notify_all();
        }
    }

    bool is_closed() const {
        return closed_.load(std::memory_order_acquire);
    }

private:
    mutable std::mutex state_m_;
    mutable std::mutex wait_m_;
    std::condition_variable cv_;
    oneapi::tbb::concurrent_bounded_queue<T> q_;
    std::atomic<std::size_t> size_{0};
    std::atomic_bool closed_{false};
};

} // namespace vio
