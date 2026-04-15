#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "keypoints/keypoint_detector.hpp"

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
    vio::IKeypointDetector& detector,
    const vio::ShiTomasiParams& params
);

cv::Mat drawTrackingVisualization(
    const cv::Mat& frame,
    const std::vector<Track>& tracks,
    const int tailLength = 15
);