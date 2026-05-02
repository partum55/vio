#include "frontend/feature_tracker.hpp"

#include <cmath>
#include <stdexcept>

namespace vio {

namespace {

bool isPointInside(const cv::Point2f& point, const cv::Size& size, int border_margin)
{
    return std::isfinite(point.x) &&
           std::isfinite(point.y) &&
           point.x >= static_cast<float>(border_margin) &&
           point.y >= static_cast<float>(border_margin) &&
           point.x < static_cast<float>(size.width - border_margin) &&
           point.y < static_cast<float>(size.height - border_margin);
}

} // namespace

FeatureTracker::FeatureTracker()
    : backend_(VisionComputeBackend::createAuto())
{
}

void FeatureTracker::setParams(const FeatureTrackerParams& params)
{
    params_ = params;
}

TrackingResult FeatureTracker::track(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const std::vector<Track>& previous_tracks
) const {
    if (prev_gray.empty() || curr_gray.empty()) {
        throw std::runtime_error("FeatureTracker::track: empty image");
    }

    std::vector<cv::Point2f> pts_prev;
    pts_prev.reserve(previous_tracks.size());

    for (const auto& track : previous_tracks) {
        pts_prev.push_back(track.pt);
    }

    std::vector<cv::Point2f> pts_curr;
    std::vector<uchar> status;
    std::vector<float> error;

    if (pts_prev.empty()) {
        return TrackingResult{};
    }

    backend_->trackPyramidalLK(
        prev_gray,
        curr_gray,
        pts_prev,
        pts_curr,
        status,
        error,
        params_.winSize,
        params_.maxLevel,
        params_.maxIters,
        params_.eps
    );

    std::vector<cv::Point2f> pts_back;
    std::vector<uchar> back_status;
    std::vector<float> back_error;
    backend_->trackPyramidalLK(
        curr_gray,
        prev_gray,
        pts_curr,
        pts_back,
        back_status,
        back_error,
        params_.winSize,
        params_.maxLevel,
        params_.maxIters,
        params_.eps
    );

    TrackingResult result;
    result.status = status;
    result.error = error;

    for (size_t i = 0; i < previous_tracks.size(); ++i) {
        if (!status[i]) {
            continue;
        }
        if (i >= pts_curr.size() || i >= pts_back.size() || i >= back_status.size() || !back_status[i]) {
            continue;
        }
        if (!isPointInside(pts_curr[i], curr_gray.size(), params_.borderMargin)) {
            continue;
        }
        if (i < error.size() && params_.maxError > 0.0f && error[i] > params_.maxError) {
            continue;
        }
        if (params_.maxForwardBackwardError > 0.0f &&
            cv::norm(pts_back[i] - pts_prev[i]) > params_.maxForwardBackwardError) {
            continue;
        }

        Track updated = previous_tracks[i];
        updated.pt = pts_curr[i];
        updated.history.push_back(pts_curr[i]);

        result.tracks.push_back(updated);
    }

    return result;
}

TrackingResult FeatureTracker::trackWithGuess(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const std::vector<Track>& previous_tracks,
    const std::vector<cv::Point2f>& initial_guess
) const {
    if (prev_gray.empty() || curr_gray.empty()) {
        throw std::runtime_error("FeatureTracker::trackWithGuess: empty image");
    }

    if (previous_tracks.size() != initial_guess.size()) {
        throw std::runtime_error("FeatureTracker::trackWithGuess: tracks and guess size mismatch");
    }

    std::vector<cv::Point2f> pts_prev;
    pts_prev.reserve(previous_tracks.size());

    for (const auto& track : previous_tracks) {
        pts_prev.push_back(track.pt);
    }

    std::vector<cv::Point2f> pts_curr;
    std::vector<uchar> status;
    std::vector<float> error;

    if (pts_prev.empty()) {
        return TrackingResult{};
    }

    backend_->trackPyramidalLKWithGuess(
        prev_gray,
        curr_gray,
        pts_prev,
        initial_guess,
        pts_curr,
        status,
        error,
        params_.winSize,
        params_.maxLevel,
        params_.maxIters,
        params_.eps
    );

    std::vector<cv::Point2f> pts_back;
    std::vector<uchar> back_status;
    std::vector<float> back_error;
    backend_->trackPyramidalLK(
        curr_gray,
        prev_gray,
        pts_curr,
        pts_back,
        back_status,
        back_error,
        params_.winSize,
        params_.maxLevel,
        params_.maxIters,
        params_.eps
    );

    TrackingResult result;
    result.status = status;
    result.error = error;

    for (size_t i = 0; i < previous_tracks.size(); ++i) {
        if (!status[i]) {
            continue;
        }
        if (i >= pts_curr.size() || i >= pts_back.size() || i >= back_status.size() || !back_status[i]) {
            continue;
        }
        if (!isPointInside(pts_curr[i], curr_gray.size(), params_.borderMargin)) {
            continue;
        }
        if (i < error.size() && params_.maxError > 0.0f && error[i] > params_.maxError) {
            continue;
        }
        if (params_.maxForwardBackwardError > 0.0f &&
            cv::norm(pts_back[i] - pts_prev[i]) > params_.maxForwardBackwardError) {
            continue;
        }

        Track updated = previous_tracks[i];
        updated.pt = pts_curr[i];
        updated.history.push_back(pts_curr[i]);

        result.tracks.push_back(updated);
    }

    return result;
}

} // namespace vio
