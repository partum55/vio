#pragma once

#include "keypoints/abc_thread_pool.hpp"

#include <opencv2/core.hpp>

namespace vio {

cv::Mat derivativeXCentral(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks);

cv::Mat derivativeYCentral(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks);

void centralDifferenceXY(
    const cv::Mat& image,
    cv::Mat& gx,
    cv::Mat& gy,
    ABCThreadPool& pool,
    int num_tasks);

} // namespace vio
