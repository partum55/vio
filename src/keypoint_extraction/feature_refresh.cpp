#include "feature_refresh.hpp"

#include <cmath>
#include <opencv2/opencv.hpp>

namespace
{
    cv::Mat makeAllowedMask(
        const cv::Size& size,
        const std::vector<cv::Point2f>& trackedPoints,
        float suppressionRadius)
    {
        cv::Mat mask(size, CV_8U, cv::Scalar(255));

        const int radius = std::max(1, static_cast<int>(std::round(suppressionRadius)));

        for (const auto& p : trackedPoints)
        {
            cv::circle(mask, p, radius, cv::Scalar(0), -1);
        }

        return mask;
    }
}

std::vector<cv::Point2f> refreshFeaturesIfNeeded(
    const cv::Mat& img,
    const std::vector<cv::Point2f>& trackedPoints,
    CustomShiTomasiDetector& detector,
    ShiTomasiParams detectorParams,
    const FeatureRefreshParams& refreshParams)
{
    if (static_cast<int>(trackedPoints.size()) >= refreshParams.minTrackedFeatures)
    {
        return trackedPoints;
    }

    const int missing = refreshParams.targetFeatures - static_cast<int>(trackedPoints.size());
    if (missing <= 0)
    {
        return trackedPoints;
    }

    detectorParams.maxCorners = missing;

    const cv::Mat allowedMask = makeAllowedMask(
        img.size(),
        trackedPoints,
        refreshParams.suppressionRadius);

    std::vector<cv::Point2f> newPoints =
        detector.detect(img, detectorParams, allowedMask);

    std::vector<cv::Point2f> result;
    result.reserve(trackedPoints.size() + newPoints.size());

    result.insert(result.end(), trackedPoints.begin(), trackedPoints.end());
    result.insert(result.end(), newPoints.begin(), newPoints.end());

    return result;
}