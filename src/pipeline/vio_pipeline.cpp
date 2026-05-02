#include "pipeline/vio_pipeline.hpp"

#include "frontend/visual_frontend.hpp"
#include "imu/imu_processor.hpp"
#include "io/data_streamer.hpp"
#include "io/dataset_loader.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vio {

namespace {

void validatePipelineParams(const VioPipelineParams& params) {
    if (params.min_tracked_points <= 0) {
        throw std::invalid_argument("VioPipelineParams::min_tracked_points must be positive");
    }
    if (params.min_landmarks_for_pnp <= 0) {
        throw std::invalid_argument("VioPipelineParams::min_landmarks_for_pnp must be positive");
    }
    if (params.min_landmarks_after_initial_triangulation <= 0) {
        throw std::invalid_argument("VioPipelineParams::min_landmarks_after_initial_triangulation must be positive");
    }
    if (params.min_shared_tracks_for_baseline <= 0) {
        throw std::invalid_argument("VioPipelineParams::min_shared_tracks_for_baseline must be positive");
    }
    if (params.min_pixel_baseline < 0.0 ||
        params.max_pixel_baseline < params.min_pixel_baseline) {
        throw std::invalid_argument("VioPipelineParams pixel baseline limits are invalid");
    }
    if (params.max_pose_baseline <= 0.0) {
        throw std::invalid_argument("VioPipelineParams::max_pose_baseline must be positive");
    }
    if (params.min_pose_baseline_translation < 0.0) {
        throw std::invalid_argument("VioPipelineParams::min_pose_baseline_translation must be non-negative");
    }
    if (params.min_pose_baseline_rotation_deg < 0.0) {
        throw std::invalid_argument("VioPipelineParams::min_pose_baseline_rotation_deg must be non-negative");
    }
    if (params.min_pnp_inliers <= 0) {
        throw std::invalid_argument("VioPipelineParams::min_pnp_inliers must be positive");
    }
}

bool validateRunConfig(const VioRunConfig& config, std::string& error) {
    if (config.imu_csv_path.empty()) {
        error = "imu_csv_path is empty";
        return false;
    }
    if (config.images_dir.empty()) {
        error = "images_dir is empty";
        return false;
    }
    if (!config.camera_intrinsics.isValid()) {
        error = "camera_intrinsics must have positive fx/fy";
        return false;
    }
    if (config.tracker_win_size <= 0 ||
        config.tracker_max_level < 0 ||
        config.tracker_max_iters <= 0 ||
        config.tracker_eps <= 0.0f) {
        error = "tracking parameters are invalid";
        return false;
    }
    if (config.triangulation.min_observations < 2 ||
        config.triangulation.min_baseline < 0.0 ||
        config.triangulation.max_baseline < config.triangulation.min_baseline ||
        config.triangulation.max_reprojection_error <= 0.0 ||
        config.triangulation.min_depth <= 0.0) {
        error = "triangulation quality parameters are invalid";
        return false;
    }
    if (config.stream_realtime && config.stream_rate <= 0.0) {
        error = "stream_rate must be positive when stream_realtime is enabled";
        return false;
    }
    return true;
}

double rotationAngleDeg(
    const Eigen::Quaterniond& q1,
    const Eigen::Quaterniond& q2
) {
    Eigen::Quaterniond dq = q1.inverse() * q2;
    dq.normalize();
    const double w = std::clamp(dq.w(), -1.0, 1.0);
    constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
    return 2.0 * std::acos(std::abs(w)) * kRadToDeg;
}

FrameState frameStateFromImuPose(
    int frame_id,
    double timestamp,
    const Pose& pose
) {
    FrameState state;
    state.frame_id = frame_id;
    state.timestamp = timestamp;
    state.q_wc = pose.q;
    state.t_wc = pose.p;
    state.v_w = pose.v;
    return state;
}

void ensureParentDirectory(const std::string& path) {
    if (path.empty()) {
        return;
    }
    const std::filesystem::path fs_path(path);
    const std::filesystem::path parent = fs_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void drainStreamedImuSamples(
    DatasetStreamer::ImuQueue& queue,
    std::optional<ImuSample>& lookahead,
    double timestamp_s
) {
    if (lookahead) {
        if (lookahead->t <= timestamp_s) {
            lookahead.reset();
        } else {
            return;
        }
    }

    while (true) {
        DatasetStreamer::ImuQueueItem sample;
        if (!queue.try_pop(sample)) {
            break;
        }
        if (!sample) {
            break;
        }
        if (sample->t > timestamp_s) {
            lookahead = std::move(*sample);
            break;
        }
    }
}

bool writeFrameStatesCsv(
    const std::string& path,
    const std::vector<TrackedFrame>& frames
) {
    if (path.empty()) {
        return true;
    }
    ensureParentDirectory(path);
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "frame_id,timestamp,qw,qx,qy,qz,tx,ty,tz,vx,vy,vz\n";
    for (const TrackedFrame& frame : frames) {
        const FrameState& s = frame.state;
        file << s.frame_id << ',' << s.timestamp << ','
             << s.q_wc.w() << ',' << s.q_wc.x() << ',' << s.q_wc.y() << ',' << s.q_wc.z() << ','
             << s.t_wc.x() << ',' << s.t_wc.y() << ',' << s.t_wc.z() << ','
             << s.v_w.x() << ',' << s.v_w.y() << ',' << s.v_w.z() << '\n';
    }
    return true;
}

bool writeObservationsCsv(
    const std::string& path,
    const std::vector<TrackedFrame>& frames
) {
    if (path.empty()) {
        return true;
    }
    ensureParentDirectory(path);
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "frame_id,timestamp,track_id,u,v,valid\n";
    for (const TrackedFrame& frame : frames) {
        for (const Observation& obs : frame.observations) {
            file << frame.state.frame_id << ',' << frame.state.timestamp << ','
                 << obs.track_id << ',' << obs.uv.x() << ',' << obs.uv.y() << ','
                 << (obs.valid ? 1 : 0) << '\n';
        }
    }
    return true;
}

bool writeLandmarksCsv(
    const std::string& path,
    const std::vector<Landmark>& landmarks
) {
    if (path.empty()) {
        return true;
    }
    ensureParentDirectory(path);
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "id,track_id,x,y,z,reprojection_error,num_observations,valid\n";
    for (const Landmark& landmark : landmarks) {
        file << landmark.id << ',' << landmark.track_id << ','
             << landmark.p_w.x() << ',' << landmark.p_w.y() << ',' << landmark.p_w.z() << ','
             << landmark.reprojection_error << ',' << landmark.num_observations << ','
             << (landmark.valid ? 1 : 0) << '\n';
    }
    return true;
}

} // namespace

VioPipeline::VioPipeline(
    const CameraIntrinsics& intrinsics,
    const VioPipelineParams& params,
    const TriangulationParams& triangulation_params,
    const PnPParams& pnp_params
)
    : intrinsics_(intrinsics),
      params_(params),
      triangulation_params_(triangulation_params),
      pnp_solver_(intrinsics, pnp_params)
{
    validatePipelineParams(params_);
    if (!intrinsics_.isValid()) {
        throw std::invalid_argument("VioPipeline requires valid camera intrinsics");
    }
}

void VioPipeline::reset() {
    stage_ = Stage::NeedPivot;
    status_ = VioStatus::NeedFirstFrame;
    pivot_ = TrackedFrame{};
    pivot_valid_ = false;
    landmark_map_.clear();
    frames_.clear();
}

void VioPipeline::processFrame(const TrackedFrame& input_frame) {
    TrackedFrame current = input_frame;

    if (stage_ == Stage::NeedPivot || !pivot_valid_) {
        initializePivot(current);
        status_ = VioStatus::NeedInitialLandmarks;
        frames_.push_back(current);
        return;
    }

    if (validObservationCount(current) < params_.min_tracked_points) {
        status_ = VioStatus::LostTracking;
        resetTrackingSegment(current);
        frames_.push_back(current);
        return;
    }

    const bool has_global_landmarks = !landmark_map_.empty();
    const bool pnp_refined =
        has_global_landmarks &&
        pnpCorrespondenceCount(current) >= params_.min_landmarks_for_pnp &&
        refinePoseWithPnP(current);

    if (!baselineReady(current)) {
        status_ = landmark_map_.empty()
            ? VioStatus::TrackingFromPivot
            : VioStatus::TrackingWithMap;
        frames_.push_back(current);
        return;
    }

    if (!baselineReasonable(current)) {
        // Do not triangulate or run PnP with an exploded parallax/pose jump.
        // Start a fresh local segment but keep the global map for later PnP.
        setPivot(current, landmark_map_.empty()
            ? Stage::NeedInitialLandmarks
            : Stage::TrackWithLandmarks);
        status_ = landmark_map_.empty()
            ? VioStatus::NeedInitialLandmarks
            : VioStatus::TrackingWithMap;
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
            status_ = VioStatus::TrackingWithMap;
        } else {
            status_ = VioStatus::NeedInitialLandmarks;
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
    if (pnp_refined) {
        (void)triangulateFromPivot(current, 0);
        setPivot(current, Stage::TrackWithLandmarks);
        status_ = VioStatus::TrackingWithMap;
        frames_.push_back(current);
        return;
    }

    // Recovery: if PnP is not possible, still try to create new landmarks from
    // a clean two-view pair. This prevents the pipeline from stalling forever.
    const bool recovered = triangulateFromPivot(current, 1);
    if (recovered) {
        setPivot(current, Stage::TrackWithLandmarks);
        status_ = VioStatus::TrackingWithMap;
    } else {
        status_ = VioStatus::TrackingFromPivot;
    }

    frames_.push_back(current);
}

const std::vector<TrackedFrame>& VioPipeline::frames() const {
    return frames_;
}

const LandmarkMap& VioPipeline::landmarks() const {
    return landmark_map_;
}

VioStatus VioPipeline::status() const {
    return status_;
}

bool VioPipeline::hasCurrentPose() const {
    return !frames_.empty();
}

FrameState VioPipeline::currentPose() const {
    return frames_.empty() ? FrameState{} : frames_.back().state;
}

VioRunResult VioPipeline::runConfigured(const VioRunConfig& config) {
    VioRunResult result;

    if (!validateRunConfig(config, result.error)) {
        result.status = VioStatus::Failed;
        return result;
    }

    try {
        DatasetLoadOptions load_options;
        load_options.imu_csv_path = config.imu_csv_path;
        load_options.images_dir = config.images_dir;
        load_options.frame_timestamps_path = config.frame_timestamps_path;
        load_options.camera_intrinsics = config.camera_intrinsics;
        Dataset dataset = loadDataset(load_options);

        if (dataset.frames.empty() || dataset.imu_samples.empty()) {
            result.status = VioStatus::Failed;
            result.error = "dataset has no frames or no IMU samples";
            return result;
        }

        ImuProcessor imu(config.gravity);
        const double init_start = dataset.imu_samples.front().t;
        if (!imu.initialize(dataset.imu_samples, init_start, 3.0)) {
            result.status = VioStatus::Failed;
            result.error = "IMU initialization failed";
            return result;
        }

        VisualFrontendParams frontend_params;
        frontend_params.tracker.winSize = config.tracker_win_size;
        frontend_params.tracker.maxLevel = config.tracker_max_level;
        frontend_params.tracker.maxIters = config.tracker_max_iters;
        frontend_params.tracker.eps = config.tracker_eps;

        VisualFrontend frontend;
        frontend.setParams(frontend_params);

        VioPipeline pipeline(
            config.camera_intrinsics,
            VioPipelineParams{},
            config.triangulation,
            PnPParams{}
        );

        StreamerConfig streamer_config;
        streamer_config.realtime = config.stream_realtime;
        streamer_config.rate = config.stream_rate;
        streamer_config.max_image_queue = config.stream_max_image_queue;

        DatasetStreamer streamer(dataset, streamer_config);
        streamer.start();

        bool have_frame = false;
        std::optional<ImuSample> imu_lookahead;
        while (true) {
            DatasetStreamer::CameraQueueItem camera_item;
            streamer.imgQueue().pop(camera_item);
            if (!camera_item) {
                break;
            }
            CameraFrame camera = std::move(*camera_item);

            if (camera.image.empty()) {
                continue;
            }

            drainStreamedImuSamples(
                streamer.imuQueue(),
                imu_lookahead,
                camera.timestamp_s
            );
            imu.propagateUntil(camera.timestamp_s);
            const Pose imu_pose = imu.getCurrentPose();
            const FrameState state = frameStateFromImuPose(
                static_cast<int>(camera.frame_index),
                camera.timestamp_s,
                imu_pose
            );

            VisualFrontendOutput output;
            if (!frontend.hasPivot()) {
                frontend.setPivot(
                    state.frame_id,
                    state.timestamp,
                    camera.image,
                    state
                );
            }

            output = frontend.track(
                state.frame_id,
                state.timestamp,
                camera.image,
                state
            );

            pipeline.processFrame(output.frame);
            if (config.frame_logger) {
                TrackedFrame logged_frame = output.frame;
                if (pipeline.hasCurrentPose()) {
                    logged_frame.state = pipeline.currentPose();
                }
                int map_correspondences = 0;
                for (const Observation& obs : logged_frame.observations) {
                    if (obs.valid && pipeline.landmarks().hasTrack(obs.track_id)) {
                        ++map_correspondences;
                    }
                }
                const bool pose_reliable =
                    map_correspondences >= VioPipelineParams{}.min_landmarks_for_pnp;
                config.frame_logger(
                    logged_frame,
                    output.tracks,
                    pipeline.landmarks().getValidLandmarks(),
                    pipeline.status(),
                    pose_reliable,
                    camera.image_path
                );
            }
            have_frame = true;

            if (!output.enough_tracks) {
                frontend.setPivot(
                    state.frame_id,
                    state.timestamp,
                    camera.image,
                    state
                );
            }
        }

        streamer.stop();

        if (!have_frame) {
            result.status = VioStatus::Failed;
            result.error = "no camera frames were streamed";
            return result;
        }

        const std::vector<Landmark> valid_landmarks =
            pipeline.landmarks().getValidLandmarks();

        if (!writeFrameStatesCsv(config.output_poses_csv, pipeline.frames())) {
            result.status = VioStatus::Failed;
            result.error = "failed to write poses CSV";
            return result;
        }
        if (!writeObservationsCsv(config.output_observations_csv, pipeline.frames())) {
            result.status = VioStatus::Failed;
            result.error = "failed to write observations CSV";
            return result;
        }
        if (!writeLandmarksCsv(config.output_landmarks_csv, valid_landmarks)) {
            result.status = VioStatus::Failed;
            result.error = "failed to write landmarks CSV";
            return result;
        }

        result.success = true;
        result.status = VioStatus::Finished;
        result.frame_count = pipeline.frames().size();
        result.landmark_count = valid_landmarks.size();
    } catch (const std::exception& ex) {
        result.success = false;
        result.status = VioStatus::Failed;
        result.error = ex.what();
    }

    return result;
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
    if (!pivot_valid_) {
        return 0.0;
    }

    const double baseline = (current.state.t_wc - pivot_.state.t_wc).norm();
    return std::isfinite(baseline) ? baseline : std::numeric_limits<double>::infinity();
}

double VioPipeline::robustPixelBaseline(const TrackedFrame& current, int* shared_count) const {
    if (shared_count != nullptr) {
        *shared_count = 0;
    }

    if (!pivot_valid_) {
        return 0.0;
    }

    std::unordered_map<int, Eigen::Vector2d> pivot_uv_by_track;
    pivot_uv_by_track.reserve(pivot_.observations.size());

    for (const Observation& obs : pivot_.observations) {
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

bool VioPipeline::poseBaselineReady(const TrackedFrame& current) const {
    if (!pivot_valid_) {
        return false;
    }

    const double translation =
        (current.state.t_wc - pivot_.state.t_wc).norm();
    const double rotation_deg =
        rotationAngleDeg(pivot_.state.q_wc, current.state.q_wc);

    return translation >= params_.min_pose_baseline_translation ||
           rotation_deg >= params_.min_pose_baseline_rotation_deg;
}

bool VioPipeline::baselineReady(const TrackedFrame& current) const {
    if (!pivot_valid_) {
        return false;
    }

    int shared = 0;
    const double pixel_bl = robustPixelBaseline(current, &shared);
    const bool pixel_ready =
        shared >= params_.min_shared_tracks_for_baseline &&
        pixel_bl >= params_.min_pixel_baseline &&
        pixel_bl <= params_.max_pixel_baseline;

    const bool pose_ready = poseBaselineReady(current);
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
    pivot_ = frame;
    pivot_valid_ = true;
    stage_ = next_stage;
}

void VioPipeline::initializePivot(const TrackedFrame& frame) {
    setPivot(frame, Stage::NeedInitialLandmarks);
}

bool VioPipeline::triangulateFromPivot(const TrackedFrame& current, int min_created) {
    if (!pivot_valid_) {
        return false;
    }

    std::vector<TrackedFrame> pair;
    pair.reserve(2);
    pair.push_back(pivot_);
    pair.push_back(current);

    const std::vector<Landmark> new_landmarks =
        triangulateLandmarks(pair, intrinsics_, triangulation_params_);

    int created = 0;
    for (const Landmark& landmark : new_landmarks) {
        if (!landmark.valid) {
            continue;
        }
        landmark_map_.addOrUpdate(
            landmark.track_id,
            landmark.p_w,
            landmark.reprojection_error,
            landmark.num_observations
        );
        ++created;
    }

    return created >= min_created && created > 0;
}

bool VioPipeline::refinePoseWithPnP(TrackedFrame& current) {
    std::vector<Eigen::Vector3d> points_3d_w;
    std::vector<Eigen::Vector2d> points_2d;
    landmark_map_.buildPnPCorrespondences(
        current.observations,
        points_3d_w,
        points_2d
    );

    const PnPResult pnp_result =
        pnp_solver_.solve(points_3d_w, points_2d, current.state);

    if (!pnp_result.success || pnp_result.inliers_count < params_.min_pnp_inliers) {
        return false;
    }

    const double jump = (pnp_result.pose.t_wc - current.state.t_wc).norm();
    if (!std::isfinite(jump)) {
        return false;
    }

    current.state = pnp_result.pose;
    return true;
}

void VioPipeline::resetTrackingSegment(const TrackedFrame& current) {
    setPivot(current, landmark_map_.empty()
        ? Stage::NeedInitialLandmarks
        : Stage::TrackWithLandmarks);
}

} // namespace vio
