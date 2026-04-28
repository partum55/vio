#pragma once

#include "core/dataset.h"
#include "imu/imu.hpp"
#include "keypoints/thread_safe_queue.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <thread>

namespace vio {

struct CameraFrame {
    std::int64_t          timestamp_ns = 0;
    double                timestamp_s  = 0.0;
    std::filesystem::path image_path;
    cv::Mat               image;   // BGR, loaded from disk by producer thread
};

struct StreamerConfig {
    bool        realtime        = false; // sleep between pushes to simulate real time
    double      rate            = 1.0;  // playback speed multiplier (realtime only)
    std::size_t max_image_queue = 8;    // 0 = unbounded; producer blocks when full
};

// Produces ImuSample and CameraFrame into two queues in strict timestamp order.
// Producer runs on a dedicated std::jthread; consumer calls imuQueue()/imgQueue().
// Both queues are closed (and deque() returns nullopt) when the dataset is exhausted
// or stop() is called.
class DatasetStreamer {
public:
    explicit DatasetStreamer(const Dataset& dataset, StreamerConfig cfg = {});
    ~DatasetStreamer() = default;   // jthread destructor: request_stop + join

    DatasetStreamer(const DatasetStreamer&)            = delete;
    DatasetStreamer& operator=(const DatasetStreamer&) = delete;

    // Spawn producer thread. Safe to call only once.
    void start();

    // Request graceful stop and join producer thread.
    void stop();

    ThreadSafeQueue<ImuSample>&   imuQueue() noexcept { return imu_queue_; }
    ThreadSafeQueue<CameraFrame>& imgQueue() noexcept { return img_queue_; }

    bool done() const noexcept { return done_.load(std::memory_order_acquire); }

private:
    void producerLoop(std::stop_token st);

    const Dataset&               dataset_;
    StreamerConfig               cfg_;
    ThreadSafeQueue<ImuSample>   imu_queue_;
    ThreadSafeQueue<CameraFrame> img_queue_;
    std::atomic_bool             done_{ false };
    std::jthread                 thread_;
};

} // namespace vio
