#pragma once

#include "abc_thread_pool.hpp"

#include <opencv2/opencv.hpp>

#include <vector>

namespace vio {

struct ShiTomasiParams
{
    int maxCorners = 500;
    double qualityLevel = 0.01;
    double minDistance = 10.0;
    int blockSize = 5;
    double gaussianSigma = 1.0;
    int nmsRadius = 2;
};

class CustomShiTomasiDetector
{
public:
    CustomShiTomasiDetector(ABCThreadPool& pool, int num_tasks);

    std::vector<cv::Point2f> detect(
        const cv::Mat& img,
        const ShiTomasiParams& params);

    std::vector<cv::Point2f> detect(
        const cv::Mat& img,
        const ShiTomasiParams& params,
        const cv::Mat& allowedMask);

    std::vector<cv::Point2f> detectGray(
        const cv::Mat& gray,
        const ShiTomasiParams& params);

    std::vector<cv::Point2f> detectGray(
        const cv::Mat& gray,
        const ShiTomasiParams& params,
        const cv::Mat& allowedMask);

private:
    struct Candidate
    {
        float score;
        int x;
        int y;
    };

    int effectiveTaskCount(const cv::Size& size) const;

    cv::Mat shiTomasiScoreImage(
        const cv::Mat& gray8,
        const ShiTomasiParams& p,
        int effective_tasks);

    cv::Mat nmsLocalMax(
        const cv::Mat& score,
        int r,
        int effective_tasks);

    std::vector<cv::Point2f> selectWithGrid(
        const cv::Mat& scoreNms,
        const ShiTomasiParams& p);

    std::vector<cv::Point2f> selectWithGrid(
        const cv::Mat& scoreNms,
        const ShiTomasiParams& p,
        const cv::Mat& allowedMask);

private:
    ABCThreadPool& pool_;
    int num_tasks_;

    cv::Mat gray32_;
    cv::Mat blur_;
    cv::Mat blur_tmp_;

    cv::Mat Ix_;
    cv::Mat Iy_;

    cv::Mat Ixx_;
    cv::Mat Iyy_;
    cv::Mat Ixy_;
    cv::Mat tensor_tmp_;

    cv::Mat score_;
};

class OpenCVShiTomasiDetector
{
public:
    std::vector<cv::Point2f> detect(
        const cv::Mat& img,
        const ShiTomasiParams& params);
};

cv::Mat toGrayU8(const cv::Mat& imgBgrOrGray);

cv::Mat drawKeypointsOnImage(
    const cv::Mat& imgBgrOrGray,
    const std::vector<cv::Point2f>& pts,
    int radius = 3,
    int thickness = 1);

} // namespace vio
