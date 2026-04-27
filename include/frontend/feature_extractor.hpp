#pragma once

#include "keypoint_extraction/shi_tomasi.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

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
    int num_threads_;
};