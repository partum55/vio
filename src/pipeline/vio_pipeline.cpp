#include "pipeline/vio_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vio {

VioPipeline::VioPipeline(
    GeometryBackend geometry,
    const VioPipelineParams& params
)
    : geometry_(std::move(geometry)),
      params_(params)
{
}

void VioPipeline::reset() {
    stage_ = Stage::NeedPivot;
    pivot_.reset();
    landmark_map_.clear();
    frames_.clear();
}

void VioPipeline::processFrame(const TrackedFrame& input_frame) {
    TrackedFrame current = input_frame;

    if (stage_ == Stage::NeedPivot || !pivot_.valid()) {
        initializePivot(current);
        frames_.push_back(current);
        return;
    }

    if (validObservationCount(current) < params_.min_tracked_points) {
        resetTrackingSegment(current);
        frames_.push_back(current);
        return;
    }

    if (!baselineReady(current)) {
        frames_.push_back(current);
        return;
    }

    if (!baselineReasonable(current)) {
        // Do not triangulate or run PnP with an exploded parallax/pose jump.
        // Start a fresh local segment but keep the global map for later PnP.
        setPivot(current, landmark_map_.empty()
            ? Stage::NeedInitialLandmarks
            : Stage::TrackWithLandmarks);
        frames_.push_back(current);
        return;
    }

    if (stage_ == Stage::NeedInitialLandmarks || landmark_map_.empty()) {
        const bool ok = triangulateFromPivot(
            current,
            params_.min_landmarks_after_initial_triangulation
        );

        if (ok) {
            setPivot(current, Stage::TrackWithLandmarks);
        }

        frames_.push_back(current);
        return;
    }

    // Assignment loop:
    // 1. keep tracking fixed global landmarks;
    // 2. when baseline/parallax is enough, solve PnP;
    // 3. accept PnP only if RANSAC is strong;
    // 4. triangulate fresh tracks from old pivot + accepted current pose;
    // 5. make current the new pivot.
    if (pnpCorrespondenceCount(current) >= params_.min_landmarks_for_pnp &&
        refinePoseWithPnP(current)) {
        (void)triangulateFromPivot(current, 0);
        setPivot(current, Stage::TrackWithLandmarks);
        frames_.push_back(current);
        return;
    }

    // Recovery: if PnP is not possible, still try to create new landmarks from
    // a clean two-view pair. This prevents the pipeline from stalling forever.
    const bool recovered = triangulateFromPivot(current, 1);
    if (recovered) {
        setPivot(current, Stage::TrackWithLandmarks);
    }

    frames_.push_back(current);
}

const std::vector<TrackedFrame>& VioPipeline::frames() const {
    return frames_;
}

const LandmarkMap& VioPipeline::landmarks() const {
    return landmark_map_;
}

int VioPipeline::validObservationCount(const TrackedFrame& frame) const {
    int valid_count = 0;
    for (const Observation& obs : frame.observations) {
        if (obs.valid && obs.track_id >= 0 && obs.uv.allFinite()) {
            ++valid_count;
        }
    }
    return valid_count;
}

int VioPipeline::pnpCorrespondenceCount(const TrackedFrame& frame) const {
    std::vector<Eigen::Vector3d> points_3d_w;
    std::vector<Eigen::Vector2d> points_2d;

    landmark_map_.buildPnPCorrespondences(
        frame.observations,
        points_3d_w,
        points_2d
    );

    return static_cast<int>(points_3d_w.size());
}

double VioPipeline::poseBaseline(const TrackedFrame& current) const {
    if (!pivot_.valid()) {
        return 0.0;
    }

    const double baseline = (current.state.t_wc - pivot_.state().t_wc).norm();
    return std::isfinite(baseline) ? baseline : std::numeric_limits<double>::infinity();
}

double VioPipeline::robustPixelBaseline(const TrackedFrame& current, int* shared_count) const {
    if (shared_count != nullptr) {
        *shared_count = 0;
    }

    if (!pivot_.valid()) {
        return 0.0;
    }

    std::unordered_map<int, Eigen::Vector2d> pivot_uv_by_track;
    pivot_uv_by_track.reserve(pivot_.frame().observations.size());

    for (const Observation& obs : pivot_.frame().observations) {
        if (obs.valid && obs.track_id >= 0 && obs.uv.allFinite()) {
            pivot_uv_by_track[obs.track_id] = obs.uv;
        }
    }

    std::vector<double> distances;
    distances.reserve(current.observations.size());

    for (const Observation& obs : current.observations) {
        if (!obs.valid || obs.track_id < 0 || !obs.uv.allFinite()) {
            continue;
        }

        const auto it = pivot_uv_by_track.find(obs.track_id);
        if (it == pivot_uv_by_track.end()) {
            continue;
        }

        const double d = (obs.uv - it->second).norm();
        if (!std::isfinite(d)) {
            continue;
        }

        // LK outliers sometimes produce coordinates hundreds/thousands of pixels
        // away. They must not define the baseline. Keep real image motion only.
        if (d >= 0.25 && d <= params_.max_pixel_baseline) {
            distances.push_back(d);
        }
    }

    if (shared_count != nullptr) {
        *shared_count = static_cast<int>(distances.size());
    }

    if (distances.empty()) {
        return 0.0;
    }

    const std::size_t mid = distances.size() / 2;
    std::nth_element(distances.begin(), distances.begin() + mid, distances.end());
    return distances[mid];
}

bool VioPipeline::baselineReady(const TrackedFrame& current) const {
    if (!pivot_.valid()) {
        return false;
    }

    int shared = 0;
    const double pixel_bl = robustPixelBaseline(current, &shared);
    const bool pixel_ready =
        shared >= params_.min_shared_tracks_for_baseline &&
        pixel_bl >= params_.min_pixel_baseline &&
        pixel_bl <= params_.max_pixel_baseline;

    const bool pose_ready = geometry_.baselineEnough(pivot_.state(), current.state);
    return pose_ready || pixel_ready;
}

bool VioPipeline::baselineReasonable(const TrackedFrame& current) const {
    const double pose_bl = poseBaseline(current);
    int shared = 0;
    const double pixel_bl = robustPixelBaseline(current, &shared);

    if (!std::isfinite(pose_bl) || pose_bl > params_.max_pose_baseline) {
        return false;
    }

    if (shared > 0 && pixel_bl > params_.max_pixel_baseline) {
        return false;
    }

    return true;
}

void VioPipeline::setPivot(const TrackedFrame& frame, Stage next_stage) {
    pivot_.set(frame);
    stage_ = next_stage;
}

void VioPipeline::initializePivot(const TrackedFrame& frame) {
    setPivot(frame, Stage::NeedInitialLandmarks);
}

bool VioPipeline::triangulateFromPivot(const TrackedFrame& current, int min_created) {
    if (!pivot_.valid()) {
        return false;
    }

    const GeometryStepResult tri_result =
        geometry_.triangulateTwoViews(
            pivot_.frame(),
            current,
            landmark_map_
        );

    return tri_result.success && tri_result.created_landmarks >= min_created;
}

bool VioPipeline::refinePoseWithPnP(TrackedFrame& current) {
    const GeometryStepResult pnp_result =
        geometry_.solvePnP(
            current,
            landmark_map_
        );

    if (!pnp_result.success || pnp_result.pnp_inliers < params_.min_pnp_inliers) {
        return false;
    }

    const double jump = (pnp_result.refined_pose.t_wc - current.state.t_wc).norm();
    if (!std::isfinite(jump) || jump > params_.max_pose_baseline) {
        return false;
    }

    current.state = pnp_result.refined_pose;
    return true;
}

void VioPipeline::resetTrackingSegment(const TrackedFrame& current) {
    setPivot(current, landmark_map_.empty()
        ? Stage::NeedInitialLandmarks
        : Stage::TrackWithLandmarks);
}

} // namespace vio
