#include "frontend/visual_frontend.hpp"

#include <stdexcept>

VisualFrontend::VisualFrontend()
{
    tracker_.setParams(params_.tracker);
}

void VisualFrontend::setParams(const VisualFrontendParams& params)
{
    params_ = params;
    tracker_.setParams(params_.tracker);
}

cv::Mat VisualFrontend::toGray(const cv::Mat& frame_bgr_or_gray) const
{
    if (frame_bgr_or_gray.empty()) {
        throw std::runtime_error("VisualFrontend::toGray: empty frame");
    }

    cv::Mat gray;

    if (frame_bgr_or_gray.channels() == 3) {
        cv::cvtColor(frame_bgr_or_gray, gray, cv::COLOR_BGR2GRAY);
    } else if (frame_bgr_or_gray.channels() == 4) {
        cv::cvtColor(frame_bgr_or_gray, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame_bgr_or_gray;
    }

    if (gray.type() != CV_8UC1) {
        cv::Mat converted;
        gray.convertTo(converted, CV_8UC1);
        return converted;
    }

    return gray;
}

std::vector<Track> VisualFrontend::makeTracks(
    const std::vector<cv::Point2f>& points
) {
    std::vector<Track> tracks;
    tracks.reserve(points.size());

    for (const auto& p : points) {
        Track t;
        t.id = next_track_id_++;
        t.pt = p;
        t.history.push_back(p);
        tracks.push_back(t);
    }

    return tracks;
}

vio::TrackedFrame VisualFrontend::makeTrackedFrame(
    const vio::FrameState& pose,
    const std::vector<Track>& tracks
) const {
    vio::TrackedFrame frame;
    frame.state = pose;

    frame.observations.reserve(tracks.size());

    for (const auto& t : tracks) {
        vio::Observation obs;
        obs.frame_id = pose.frame_id;
        obs.track_id = t.id;
        obs.uv = Eigen::Vector2d(t.pt.x, t.pt.y);
        obs.valid = true;
        frame.observations.push_back(obs);
    }

    return frame;
}

void VisualFrontend::setPivot(
    int frame_id,
    double timestamp,
    const cv::Mat& frame_bgr_or_gray,
    const vio::FrameState& pose
) {
    cv::Mat gray = toGray(frame_bgr_or_gray);

    ShiTomasiParams detector_params;
    detector_params.maxCorners = params_.initialFeatures;
    detector_params.qualityLevel = params_.refresh.qualityLevel;
    detector_params.minDistance = params_.refresh.minDistance;
    detector_params.blockSize = 5;
    detector_params.gaussianSigma = 1.0;
    detector_params.nmsRadius = 2;

    const std::vector<cv::Point2f> points =
        extractor_.extract(gray, detector_params);

    if (points.empty()) {
        throw std::runtime_error("VisualFrontend::setPivot: no features found");
    }

    active_tracks_ = makeTracks(points);

    pivot_.set(
        frame_id,
        timestamp,
        gray,
        pose,
        active_tracks_
    );
}

VisualFrontendOutput VisualFrontend::track(
    int frame_id,
    double timestamp,
    const cv::Mat& frame_bgr_or_gray,
    const vio::FrameState& pose
) {
    if (!pivot_.isValid()) {
        throw std::runtime_error("VisualFrontend::track: pivot is not set");
    }

    cv::Mat curr_gray = toGray(frame_bgr_or_gray);

    TrackingResult tracked = tracker_.track(
        pivot_.gray(),
        curr_gray,
        active_tracks_
    );

    active_tracks_ = tracked.tracks;

    refreshTracksIfNeeded(
        curr_gray,
        active_tracks_,
        next_track_id_,
        params_.refresh
    );

    VisualFrontendOutput output;
    output.tracks = active_tracks_;
    output.frame = makeTrackedFrame(pose, active_tracks_);
    output.enough_tracks = hasEnoughTracks();

    return output;
}

VisualFrontendOutput VisualFrontend::trackWithGuess(
    int frame_id,
    double timestamp,
    const cv::Mat& frame_bgr_or_gray,
    const vio::FrameState& pose,
    const std::vector<cv::Point2f>& initial_guess
) {
    if (!pivot_.isValid()) {
        throw std::runtime_error("VisualFrontend::trackWithGuess: pivot is not set");
    }

    cv::Mat curr_gray = toGray(frame_bgr_or_gray);

    TrackingResult tracked = tracker_.trackWithGuess(
        pivot_.gray(),
        curr_gray,
        active_tracks_,
        initial_guess
    );

    active_tracks_ = tracked.tracks;

    refreshTracksIfNeeded(
        curr_gray,
        active_tracks_,
        next_track_id_,
        params_.refresh
    );

    VisualFrontendOutput output;
    output.tracks = active_tracks_;
    output.frame = makeTrackedFrame(pose, active_tracks_);
    output.enough_tracks = hasEnoughTracks();

    return output;
}

bool VisualFrontend::hasPivot() const
{
    return pivot_.isValid();
}

bool VisualFrontend::hasEnoughTracks() const
{
    return static_cast<int>(active_tracks_.size()) >= params_.minTrackedFeatures;
}

const PivotFrame& VisualFrontend::pivot() const
{
    return pivot_;
}

const std::vector<Track>& VisualFrontend::activeTracks() const
{
    return active_tracks_;
}

void VisualFrontend::setPivotWithTracks(
    int frame_id,
    double timestamp,
    const cv::Mat& frame_bgr_or_gray,
    const vio::FrameState& pose,
    const std::vector<Track>& tracks
) {
    if (tracks.empty()) {
        throw std::runtime_error("VisualFrontend::setPivotWithTracks: empty tracks");
    }

    cv::Mat gray = toGray(frame_bgr_or_gray);

    active_tracks_ = tracks;

    int max_id = -1;
    for (const auto& t : active_tracks_) {
        max_id = std::max(max_id, t.id);
    }
    next_track_id_ = max_id + 1;

    pivot_.set(
        frame_id,
        timestamp,
        gray,
        pose,
        active_tracks_
    );
}