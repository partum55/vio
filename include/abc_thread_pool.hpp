#pragma once

#include <functional>
#include <future>

class ABCThreadPool {
public:
    virtual ~ABCThreadPool() = default;

    virtual std::future<void> submit_task(std::function<void()> task) = 0;
    virtual bool run_pending_task() = 0;
};
