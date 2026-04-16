#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct Track
{
    int id;
    cv::Point2f pt;
    std::vector<cv::Point2f> history;
};

float pointDistance(const cv::Point2f& a, const cv::Point2f& b);

bool isFarFromExisting(
    const cv::Point2f& p,
    const std::vector<Track>& tracks,
    const float minDist
);

void addNewTracks(
    const cv::Mat& gray,
    std::vector<Track>& tracks,
    int& nextTrackId,
    const int maxCorners = 100,
    const double qualityLevel = 0.01,
    const double minDistance = 10.0
);

cv::Mat drawTrackingVisualization(
    const cv::Mat& frame,
    const std::vector<Track>& tracks,
    const int tailLength = 15
);