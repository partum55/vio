#pragma once

#include "abc_thread_pool.hpp"
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>

namespace vio {

template <typename Func>
void parallel_for_rows(
    ABCThreadPool& pool,
    int rows,
    int num_tasks,
    Func&& func)
{
    (void)pool;
    if (rows <= 0) {
        return;
    }

    const int tasks = std::max(1, std::min(num_tasks, rows));
    if (tasks == 1) {
        func(0, rows);
        return;
    }

    const int chunk = (rows + tasks - 1) / tasks;
    oneapi::tbb::global_control gc(
        oneapi::tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(tasks)
    );
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<int>(0, rows, chunk),
        [&](const oneapi::tbb::blocked_range<int>& range) {
            func(range.begin(), range.end());
        }
    );
}

} // namespace vio
