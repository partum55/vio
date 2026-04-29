#pragma once

#include "core/types.hpp"
#include "frontend/vision_compute_backend.hpp"

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace vio {

struct FeatureTrackerParams {
    int winSize = 9;
    int maxLevel = 3;
    int maxIters = 10;
    float eps = 1e-3f;
};

struct TrackingResult {
    std::vector<Track> tracks;
    std::vector<uchar> status;
    std::vector<float> error;
};

class FeatureTracker {
public:
    FeatureTracker();

    void setParams(const FeatureTrackerParams& params);

    TrackingResult track(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const std::vector<Track>& previous_tracks
    ) const;

    TrackingResult trackWithGuess(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const std::vector<Track>& previous_tracks,
        const std::vector<cv::Point2f>& initial_guess
    ) const;

private:
    FeatureTrackerParams params_;
    std::shared_ptr<VisionComputeBackend> backend_;
};

} // namespace vio
