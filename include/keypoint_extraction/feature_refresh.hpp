#pragma once

#include "shi_tomasi.hpp"

struct FeatureRefreshParams
{
    int minTrackedFeatures = 200;
    int targetFeatures = 500;
    float suppressionRadius = 10.0f;
};

std::vector<cv::Point2f> refreshFeaturesIfNeeded(
    const cv::Mat& img,
    const std::vector<cv::Point2f>& trackedPoints,
    CustomShiTomasiDetector& detector,
    ShiTomasiParams detectorParams,
    const FeatureRefreshParams& refreshParams);