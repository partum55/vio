#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace vio {

void trackPoints(const cv::Mat& prevGray,
                 const cv::Mat& currGray,
                 const std::vector<cv::Point2f>& pts0,
                 std::vector<cv::Point2f>& pts1,
                 std::vector<uchar>& status,
                 std::vector<float>& err,
                 int winSize = 9,
                 int maxLevel = 3,
                 int maxIters = 10,
                 float eps = 1e-3f);

} // namespace vio
