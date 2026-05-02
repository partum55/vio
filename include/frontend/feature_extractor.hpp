#pragma once

#include "frontend/vision_compute_backend.hpp"

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace vio {

struct ShiTomasiParams {
    int maxCorners = 500;
    double qualityLevel = 0.01;
    double minDistance = 10.0;
    int blockSize = 5;
    double gaussianSigma = 1.0;
    int nmsRadius = 2;
};

class FeatureExtractor {
public:
    FeatureExtractor();

    std::vector<cv::Point2f> extract(
        const cv::Mat& gray,
        const ShiTomasiParams& params
    );

    std::vector<cv::Point2f> extract(
        const cv::Mat& gray,
        const ShiTomasiParams& params,
        const cv::Mat& mask
    );

private:
    cv::Mat scoreImage(
        const cv::Mat& gray,
        const ShiTomasiParams& params
    ) const;

    std::vector<cv::Point2f> selectFeatures(
        const cv::Mat& score,
        const ShiTomasiParams& params,
        const cv::Mat& mask
    ) const;

private:
    std::shared_ptr<VisionComputeBackend> backend_;
};

} // namespace vio
