#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
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

    bool enque(const T& value) {
        {
            std::lock_guard<std::mutex> lg(m_);
            if (closed_) {
                return false;
            }
            q_.push_back(value);
        }
        cv_.notify_one();
        return true;
    }

    bool enque(T&& value) {
        {
            std::lock_guard<std::mutex> lg(m_);
            if (closed_) {
                return false;
            }
            q_.push_back(std::move(value));
        }
        cv_.notify_one();
        return true;
    }

    template <class... Args>
    bool emplace(Args&&... args) {
        {
            std::lock_guard<std::mutex> lg(m_);
            if (closed_) {
                return false;
            }
            q_.emplace_back(std::forward<Args>(args)...);
        }
        cv_.notify_one();
        return true;
    }

    std::optional<T> deque() {
        std::unique_lock<std::mutex> ul(m_);
        cv_.wait(ul, [this] {
            return closed_ || !q_.empty();
        });

        if (q_.empty()) {
            return std::nullopt;
        }

        T value = std::move(q_.front());
        q_.pop_front();
        return value;
    }

    std::optional<T> try_deque() {
        std::lock_guard<std::mutex> lg(m_);
        if (q_.empty()) {
            return std::nullopt;
        }

        T value = std::move(q_.front());
        q_.pop_front();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lg(m_);
        return q_.empty();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lg(m_);
        return q_.size();
    }

    void close() noexcept {
        bool need_notify = false;
        {
            std::lock_guard<std::mutex> lg(m_);
            if (!closed_) {
                closed_ = true;
                need_notify = true;
            }
        }
        if (need_notify) {
            cv_.notify_all();
        }
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lg(m_);
        return closed_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<T> q_;
    bool closed_{false};
};

} // namespace vio
