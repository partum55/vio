#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "tracking/tracking_vis.hpp"
#include "keypoints/keypoint_detector.hpp"

struct FeatureRefreshParams
{
    int minTrackedFeatures = 50;
    int targetFeatures = 100;
    float suppressionRadius = 10.0f;
    double qualityLevel = 0.01;
    double minDistance = 10.0;
};

cv::Mat makeAllowedMask(
    const cv::Size& size,
    const std::vector<Track>& tracks,
    float suppressionRadius
);

void refreshTracksIfNeeded(
    const cv::Mat& gray,
    std::vector<Track>& tracks,
    int& nextTrackId,
    const FeatureRefreshParams& params,
    vio::IKeypointDetector& detector
);