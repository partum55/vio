#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct ShiTomasiParams
{
    int maxCorners = 1000;
    double qualityLevel = 0.01; // threshold relative to max score
    double minDistance = 8.0;
    int blockSize = 5;          // window for structure tensor
    int sobelKSize = 3;
    double gaussianSigma = 1.0;
    int nmsRadius = 2;          // local maxima radius (>=1)'
};

struct Candidate
{
    float score;
    int x;
    int y;
};

std::vector<cv::Point2f> extractShiTomasiKeypoints(const cv::Mat &img, const ShiTomasiParams &p);

cv::Mat toGrayU8(const cv::Mat &imgBgrOrGray);

cv::Mat drawKeypointsOnImage(const cv::Mat &imgBgrOrGray,
                             const std::vector<cv::Point2f> &pts,
                             int radius = 2,
                             int thickness = -1);
