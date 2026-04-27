#pragma once

#include "core/tracked_frame.hpp"
#include "frontend/feature_extractor.hpp"
#include "frontend/feature_tracker.hpp"
#include "frontend/pivot_frame.hpp"
#include "tracking/feature_refresh.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

struct VisualFrontendParams {
    FeatureTrackerParams tracker;
    FeatureRefreshParams refresh;

    int initialFeatures = 100;
    int minTrackedFeatures = 50;
};

struct VisualFrontendOutput {
    vio::TrackedFrame frame;
    std::vector<Track> tracks;
    bool enough_tracks = false;
};

class VisualFrontend {
public:
    VisualFrontend();

    void setParams(const VisualFrontendParams& params);

    void setPivot(
        int frame_id,
        double timestamp,
        const cv::Mat& frame_bgr_or_gray,
        const vio::FrameState& pose
    );

    VisualFrontendOutput track(
        int frame_id,
        double timestamp,
        const cv::Mat& frame_bgr_or_gray,
        const vio::FrameState& pose
    );

    VisualFrontendOutput trackWithGuess(
        int frame_id,
        double timestamp,
        const cv::Mat& frame_bgr_or_gray,
        const vio::FrameState& pose,
        const std::vector<cv::Point2f>& initial_guess
    );

    void setPivotWithTracks(
        int frame_id,
        double timestamp,
        const cv::Mat& frame_bgr_or_gray,
        const vio::FrameState& pose,
        const std::vector<Track>& tracks
    );

    bool hasPivot() const;
    bool hasEnoughTracks() const;

    const PivotFrame& pivot() const;
    const std::vector<Track>& activeTracks() const;

private:
    cv::Mat toGray(const cv::Mat& frame_bgr_or_gray) const;

    std::vector<Track> makeTracks(
        const std::vector<cv::Point2f>& points
    );

    vio::TrackedFrame makeTrackedFrame(
        const vio::FrameState& pose,
        const std::vector<Track>& tracks
    ) const;

private:
    VisualFrontendParams params_;

    FeatureExtractor extractor_;
    FeatureTracker tracker_;
    PivotFrame pivot_;

    std::vector<Track> active_tracks_;
    int next_track_id_ = 0;
};