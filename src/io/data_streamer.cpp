#include "io/data_streamer.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstddef>
#include <thread>

namespace vio {

DatasetStreamer::DatasetStreamer(const Dataset& dataset, StreamerConfig cfg)
    : dataset_(dataset), cfg_(cfg)
{
    if (cfg_.max_image_queue > 0) {
        img_queue_.set_capacity(cfg_.max_image_queue);
    }
}

void DatasetStreamer::start() {
    if (thread_.joinable()) return;
    done_.store(false, std::memory_order_release);
    thread_ = std::jthread([this](std::stop_token st) { producerLoop(st); });
}

void DatasetStreamer::stop() {
    thread_.request_stop();
    if (thread_.joinable()) thread_.join();
}

void DatasetStreamer::producerLoop(std::stop_token st) {
    using Clock = std::chrono::steady_clock;

    const auto& frames      = dataset_.frames;
    const auto& imu_samples = dataset_.imu_samples;

    // Compute real-time anchor from the earliest event in either stream.
    double data_start = 0.0;
    if (!frames.empty() && !imu_samples.empty())
        data_start = std::min(frames.front().timestamp_s, imu_samples.front().t);
    else if (!frames.empty())
        data_start = frames.front().timestamp_s;
    else if (!imu_samples.empty())
        data_start = imu_samples.front().t;

    const auto wall_start = Clock::now();

    // Two-pointer merge: interleave IMU and camera events in timestamp order.
    // IMU samples with equal timestamp to a camera frame come first.
    std::size_t fi = 0;
    std::size_t ii = 0;

    while ((fi < frames.size() || ii < imu_samples.size()) && !st.stop_requested()) {
        const bool take_imu =
            (ii < imu_samples.size()) &&
            (fi >= frames.size() || imu_samples[ii].t <= frames[fi].timestamp_s);

        const double ev_time = take_imu ? imu_samples[ii].t : frames[fi].timestamp_s;

        // Optional real-time sleep.
        if (cfg_.realtime && cfg_.rate > 0.0) {
            const double elapsed_s = (ev_time - data_start) / cfg_.rate;
            const auto target = wall_start + std::chrono::duration<double>(elapsed_s);
            std::this_thread::sleep_until(target);
            if (st.stop_requested()) break;
        }

        if (take_imu) {
            imu_queue_.push(ImuQueueItem{imu_samples[ii]});
            ++ii;
        } else {
            const DatasetFrame& df = frames[fi];
            cv::Mat image = cv::imread(df.image_path.string(), cv::IMREAD_COLOR);
            if (!image.empty()) {
                CameraFrame cf;
                cf.timestamp_ns = df.timestamp_ns;
                cf.timestamp_s  = df.timestamp_s;
                cf.frame_index  = df.frame_index;
                cf.image_path   = df.image_path;
                cf.image        = std::move(image);
                img_queue_.push(CameraQueueItem{std::move(cf)});
            }
            ++fi;
        }
    }

    imu_queue_.push(ImuQueueItem{std::nullopt});
    img_queue_.push(CameraQueueItem{std::nullopt});
    done_.store(true, std::memory_order_release);
}

} // namespace vio
