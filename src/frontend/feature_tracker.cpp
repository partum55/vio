#include "frontend/feature_tracker.hpp"

#include "tracking/lk_tracker.hpp"

#include <stdexcept>

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

    trackPointsPyramidalLK(
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

    TrackingResult result;
    result.status = status;
    result.error = error;

    for (size_t i = 0; i < previous_tracks.size(); ++i) {
        if (!status[i]) {
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

    trackPointsPyramidalLKWithGuess(
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

    TrackingResult result;
    result.status = status;
    result.error = error;

    for (size_t i = 0; i < previous_tracks.size(); ++i) {
        if (!status[i]) {
            continue;
        }

        Track updated = previous_tracks[i];
        updated.pt = pts_curr[i];
        updated.history.push_back(pts_curr[i]);

        result.tracks.push_back(updated);
    }

    return result;
}