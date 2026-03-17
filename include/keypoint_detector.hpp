#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct ShiTomasiParams
{
    int maxCorners = 1000;
    double qualityLevel = 0.01;
    double minDistance = 8.0;
    int blockSize = 5;
    double gaussianSigma = 1.0;
    int nmsRadius = 2;
};

class IKeypointDetector
{
public:
    virtual ~IKeypointDetector() = default;

    virtual std::vector<cv::Point2f> detect(
        const cv::Mat &img,
        const ShiTomasiParams &params) = 0;
};
