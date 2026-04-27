#include "frontend/feature_extractor.hpp"

#include "keypoint_extraction/tpool_default.hpp"

#include <thread>
#include <algorithm>
#include <stdexcept>

FeatureExtractor::FeatureExtractor()
{
    num_threads_ = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency())
    );
}

std::vector<cv::Point2f> FeatureExtractor::extract(
    const cv::Mat& gray,
    const ShiTomasiParams& params
) {
    if (gray.empty()) {
        throw std::runtime_error("FeatureExtractor::extract: empty image");
    }

    if (gray.type() != CV_8UC1) {
        throw std::runtime_error("FeatureExtractor::extract: expected CV_8UC1 grayscale image");
    }

    ThreadPool pool(num_threads_);
    CustomShiTomasiDetector detector(pool, num_threads_);

    return detector.detectGray(gray, params);
}

std::vector<cv::Point2f> FeatureExtractor::extract(
    const cv::Mat& gray,
    const ShiTomasiParams& params,
    const cv::Mat& mask
) {
    if (gray.empty()) {
        throw std::runtime_error("FeatureExtractor::extract: empty image");
    }

    if (gray.type() != CV_8UC1) {
        throw std::runtime_error("FeatureExtractor::extract: expected CV_8UC1 grayscale image");
    }

    if (!mask.empty() && (mask.type() != CV_8UC1 || mask.size() != gray.size())) {
        throw std::runtime_error("FeatureExtractor::extract: invalid mask");
    }

    ThreadPool pool(num_threads_);
    CustomShiTomasiDetector detector(pool, num_threads_);

    return detector.detectGray(gray, params, mask);
}