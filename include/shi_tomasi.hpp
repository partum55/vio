#pragma once

#include "abc_thread_pool.hpp"
#include "keypoint_detector.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

class CustomShiTomasiDetector : public IKeypointDetector
{
public:
    CustomShiTomasiDetector(ABCThreadPool& pool, int num_tasks);

    std::vector<cv::Point2f> detect(
        const cv::Mat &img,
        const ShiTomasiParams &params) override;

private:
    struct Candidate
    {
        float score;
        int x;
        int y;
    };

    cv::Mat shiTomasiScoreImage(
        const cv::Mat &gray8,
        const ShiTomasiParams &p);

    cv::Mat nmsLocalMax(const cv::Mat &score, int r);

    std::vector<cv::Point2f> selectWithGrid(
        const cv::Mat &scoreNms,
        const ShiTomasiParams &p);

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
    cv::Mat dilated_;
};

class OpenCVShiTomasiDetector : public IKeypointDetector
{
public:
    std::vector<cv::Point2f> detect(
        const cv::Mat &img,
        const ShiTomasiParams &params) override;
};

cv::Mat toGrayU8(const cv::Mat &imgBgrOrGray);

cv::Mat drawKeypointsOnImage(
    const cv::Mat &imgBgrOrGray,
    const std::vector<cv::Point2f> &pts,
    int radius = 3,
    int thickness = 1);