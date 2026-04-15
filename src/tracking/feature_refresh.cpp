#include "tracking/feature_refresh.hpp"

#include "keypoints/keypoint_detector.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

cv::Mat makeAllowedMask(
    const cv::Size& size,
    const std::vector<Track>& tracks,
    float suppressionRadius
) {
    cv::Mat mask(size, CV_8U, cv::Scalar(255));

    const int radius = std::max(1, static_cast<int>(std::round(suppressionRadius)));

    for (const auto& t : tracks) {
        cv::circle(mask, t.pt, radius, cv::Scalar(0), -1);
    }

    return mask;
}

void refreshTracksIfNeeded(
    const cv::Mat& gray,
    std::vector<Track>& tracks,
    int& nextTrackId,
    const FeatureRefreshParams& params,
    vio::IKeypointDetector& detector
) {
    if (gray.empty()) {
        throw std::runtime_error("refreshTracksIfNeeded: empty image");
    }

    if (gray.type() != CV_8UC1) {
        throw std::runtime_error("refreshTracksIfNeeded: expected CV_8UC1 grayscale image");
    }

    if (static_cast<int>(tracks.size()) >= params.minTrackedFeatures) {
        return;
    }

    const int missing = params.targetFeatures - static_cast<int>(tracks.size());
    if (missing <= 0) {
        return;
    }

    vio::ShiTomasiParams shi_params;
    shi_params.maxCorners = missing * 3;
    shi_params.qualityLevel = params.qualityLevel;
    shi_params.minDistance = params.minDistance;

    const std::vector<cv::Point2f> detected = detector.detect(gray, shi_params);

    int added = 0;
    for (const auto& p : detected) {
        if (added >= missing) break;
        if (isFarFromExisting(p, tracks, static_cast<float>(params.suppressionRadius))) {
            Track t;
            t.id = nextTrackId++;
            t.pt = p;
            t.history.push_back(p);
            tracks.push_back(t);
            ++added;
        }
    }
}