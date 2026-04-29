#pragma once

#include "core/types.hpp"
#include "frontend/feature_extractor.hpp"
#include "frontend/feature_tracker.hpp"
#include "frontend/pivot_frame.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace vio {

struct FeatureRefreshParams {
    int minTrackedFeatures = 50;
    int targetFeatures = 100;
    float suppressionRadius = 10.0f;
    double qualityLevel = 0.01;
    double minDistance = 10.0;
};

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

    void refreshTracksIfNeeded(const cv::Mat& gray);

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

} // namespace vio
