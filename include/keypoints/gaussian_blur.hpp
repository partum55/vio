#pragma once

#include "keypoints/abc_thread_pool.hpp"

#include <opencv2/core.hpp>

namespace vio {

cv::Mat convolveHorizontalReplicate(
    const cv::Mat& image,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks);

cv::Mat convolveVerticalReplicate(
    const cv::Mat& image,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks);

cv::Mat gaussianBlurCustom(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks);

} // namespace vio
