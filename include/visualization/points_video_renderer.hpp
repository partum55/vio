#pragma once

#include "core/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace vio {

cv::Mat drawPointsVideoFrame(
    const cv::Mat& frame,
    const std::vector<Track>& tracks,
    int tail_length = 15,
    const FrameState* state = nullptr
);

} // namespace vio
