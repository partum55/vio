#include "feature_refresh.hpp"

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
    const FeatureRefreshParams& params
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

    const cv::Mat mask = makeAllowedMask(
        gray.size(),
        tracks,
        params.suppressionRadius
    );

    std::vector<cv::Point2f> detected;
    cv::goodFeaturesToTrack(
        gray,
        detected,
        missing,
        params.qualityLevel,
        params.minDistance,
        mask
    );

    for (const auto& p : detected) {
        Track t;
        t.id = nextTrackId++;
        t.pt = p;
        t.history.push_back(p);
        tracks.push_back(t);
    }
}