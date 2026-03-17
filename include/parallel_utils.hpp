#pragma once

#include "abc_thread_pool.hpp"

#include <algorithm>
#include <future>
#include <vector>

template <typename Func>
void parallel_for_rows(
    ABCThreadPool& pool,
    int rows,
    int num_tasks,
    Func&& func)
{
    if (rows <= 0) {
        return;
    }

    const int tasks = std::max(1, std::min(num_tasks, rows));
    const int chunk = (rows + tasks - 1) / tasks;

    std::vector<std::future<void>> futures;
    futures.reserve(tasks);

    for (int t = 0; t < tasks; ++t) {
        const int y0 = t * chunk;
        const int y1 = std::min(rows, y0 + chunk);

        if (y0 >= y1) {
            break;
        }

        futures.emplace_back(
            pool.submit_task([y0, y1, &func]() {
                func(y0, y1);
            })
        );
    }

    for (auto& f : futures) {
        f.get();
    }
}
