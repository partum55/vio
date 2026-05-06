#include "pipeline/vio_pipeline.hpp"

#include "frontend/visual_frontend.hpp"
#include "imu/imu_preintegration.hpp"
#include "imu/imu_processor.hpp"
#include "io/data_streamer.hpp"
#include "io/dataset_loader.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

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
#include <unordered_set>
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
    if (params.min_epipolar_inliers <= 0) {
        throw std::invalid_argument("VioPipelineParams::min_epipolar_inliers must be positive");
    }
    if (params.epipolar_ransac_threshold_px <= 0.0 ||
        params.epipolar_confidence <= 0.0 ||
        params.epipolar_confidence >= 1.0) {
        throw std::invalid_argument("VioPipelineParams epipolar RANSAC parameters are invalid");
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
    if (!config.images_are_undistorted) {
        error = "distorted input images are not supported by this pipeline yet";
        return false;
    }
    if (config.imu_init_duration_sec <= 0.0) {
        error = "imu_init_duration_sec must be positive";
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

bool isFiniteFrameState(const FrameState& state) {
    return state.q_wc.coeffs().allFinite() &&
           state.t_wc.allFinite() &&
           state.v_w.allFinite() &&
           state.a_w.allFinite();
}

double maxReasonableFrameStep(double t0, double t1) {
    const double dt = t1 - t0;
    if (!std::isfinite(dt) || dt <= 0.0) {
        return 0.20;
    }
    return std::clamp(3.0 * dt, 0.08, 0.20);
}

FrameState frameStateFromBodyPose(
    int frame_id,
    double timestamp,
    const Pose& pose,
    const CameraImuExtrinsics& extrinsics
) {
    FrameState state;
    state.frame_id = frame_id;
    state.timestamp = timestamp;
    const Eigen::Matrix3d R_wb = pose.q.toRotationMatrix();
    const Eigen::Matrix3d R_wc = R_wb * extrinsics.R_BC();
    state.q_wc = Eigen::Quaterniond(R_wc).normalized();
    state.t_wc = pose.p + R_wb * extrinsics.t_BC();
    state.v_w = pose.v;
    state.a_w = pose.a;
    return state;
}

CameraIntrinsics intrinsicsFromCalibration(
    const CameraCalibration& calibration,
    const CameraIntrinsics& fallback
) {
    if (calibration.fx <= 0.0 || calibration.fy <= 0.0) {
        return fallback;
    }

    CameraIntrinsics intrinsics;
    intrinsics.fx = calibration.fx;
    intrinsics.fy = calibration.fy;
    intrinsics.cx = calibration.cx;
    intrinsics.cy = calibration.cy;
    intrinsics.width = calibration.width;
    intrinsics.height = calibration.height;
    return intrinsics;
}

Eigen::Matrix4d selectCameraExtrinsics(
    const VioRunConfig& config,
    const Dataset& dataset
) {
    const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    if (!config.T_BS.isApprox(identity)) {
        return config.T_BS;
    }
    if (config.use_dataset_calibration &&
        !dataset.camera.T_BS.isApprox(identity)) {
        return dataset.camera.T_BS;
    }
    return config.T_BS;
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

    file << "frame_id,timestamp,qw,qx,qy,qz,tx,ty,tz,vx,vy,vz,ax,ay,az\n";
    file << std::fixed << std::setprecision(9);
    for (const TrackedFrame& frame : frames) {
        const FrameState& s = frame.state;
        file << s.frame_id << ',' << s.timestamp << ','
             << s.q_wc.w() << ',' << s.q_wc.x() << ',' << s.q_wc.y() << ',' << s.q_wc.z() << ','
             << s.t_wc.x() << ',' << s.t_wc.y() << ',' << s.t_wc.z() << ','
             << s.v_w.x() << ',' << s.v_w.y() << ',' << s.v_w.z() << ','
             << s.a_w.x() << ',' << s.a_w.y() << ',' << s.a_w.z() << '\n';
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
    const PnPParams& pnp_params,
    const CameraImuExtrinsics& extrinsics,
    const Eigen::Vector3d& gravity,
    const SlidingWindowEstimatorParams& estimator_params
)
    : intrinsics_(intrinsics),
      params_(params),
      triangulation_params_(triangulation_params),
      pnp_solver_(intrinsics, pnp_params),
      extrinsics_(extrinsics),
      estimator_(intrinsics, extrinsics, estimator_params, gravity)
{
    validatePipelineParams(params_);
    if (!intrinsics_.isValid()) {
        throw std::invalid_argument("VioPipeline requires valid camera intrinsics");
    }
}

void VioPipeline::setImuProcessor(ImuProcessor* imu_processor)
{
    imu_processor_ = imu_processor;
}

void VioPipeline::reset() {
    stage_ = Stage::NeedPivot;
    status_ = VioStatus::NeedFirstFrame;
    pivot_ = TrackedFrame{};
    pivot_valid_ = false;
    landmark_map_.clear();
    frames_.clear();
    imu_preintegrations_.clear();
    estimator_.reset();
    estimator_status_ = SlidingWindowEstimatorStatus::Uninitialized;
}

void VioPipeline::processFrame(const TrackedFrame& input_frame) {
    processFrame(input_frame, std::nullopt);
}

void VioPipeline::processFrame(
    const TrackedFrame& input_frame,
    const std::optional<PreintegratedImuMeasurement>& imu_preintegration
) {
    TrackedFrame current = input_frame;

    if (imu_preintegration && imu_preintegration->valid) {
        imu_preintegrations_.push_back(*imu_preintegration);
    }

    if (stage_ == Stage::NeedPivot || !pivot_valid_) {
        initializePivot(current);
        status_ = VioStatus::NeedInitialLandmarks;
        finalizeFrame(current, imu_preintegration);
        return;
    }

    seedMetricMotionPrior(current);

    if (validObservationCount(current) < params_.min_tracked_points) {
        status_ = VioStatus::LostTracking;
        resetTrackingSegment(current);
        finalizeFrame(current, imu_preintegration);
        return;
    }

    const bool has_global_landmarks = !landmark_map_.empty();
    const bool pnp_refined =
        has_global_landmarks &&
        pnpCorrespondenceCount(current) >= params_.min_landmarks_for_pnp &&
        refinePoseWithPnP(current);

    int shared = 0;
    (void)robustPixelBaseline(current, &shared);

    if (shared < params_.min_shared_tracks_for_baseline) {
        // Lost too many tracks from the old pivot. Pick a new one.
        setPivot(current, landmark_map_.empty() 
            ? Stage::NeedInitialLandmarks 
            : Stage::TrackWithLandmarks);
        finalizeFrame(current, imu_preintegration);
        return;
    }

    if (!baselineReady(current)) {
        status_ = landmark_map_.empty()
            ? VioStatus::TrackingFromPivot
            : VioStatus::TrackingWithMap;
        finalizeFrame(current, imu_preintegration);
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
        finalizeFrame(current, imu_preintegration);
        return;
    }

    if (stage_ == Stage::NeedInitialLandmarks || landmark_map_.empty()) {
        const int min_created = landmark_map_.empty()
            ? params_.min_landmarks_after_initial_triangulation
            : 1;
        const bool ok = triangulateFromPivot(
            current,
            min_created
        );

        if (ok && static_cast<int>(landmark_map_.size()) >= params_.min_landmarks_for_pnp) {
            setPivot(current, Stage::TrackWithLandmarks);
            status_ = VioStatus::TrackingWithMap;
        } else {
            // Keep the same pivot to accumulate more parallax
            status_ = VioStatus::NeedInitialLandmarks;
        }

        finalizeFrame(current, imu_preintegration);
        return;
    }

    // Always try to triangulate new points from the pivot if we have enough parallax
    bool newly_triangulated = false;
    if (baselineReady(current)) {
        newly_triangulated = triangulateFromPivot(current, 0);
    }

    if (pnp_refined || newly_triangulated) {
        setPivot(current, Stage::TrackWithLandmarks);
    }
    
    status_ = landmark_map_.empty()
        ? VioStatus::TrackingFromPivot
        : VioStatus::TrackingWithMap;
    finalizeFrame(current, imu_preintegration);
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

SlidingWindowEstimatorStatus VioPipeline::estimatorStatus() const {
    return estimator_status_;
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

        const CameraIntrinsics runtime_intrinsics =
            config.use_dataset_calibration
                ? intrinsicsFromCalibration(dataset.camera, config.camera_intrinsics)
                : config.camera_intrinsics;

        CameraImuExtrinsics extrinsics;
        extrinsics.T_BS = selectCameraExtrinsics(config, dataset);
        std::cout << "[VIO] Extrinsics T_BS:\n" << extrinsics.T_BS << std::endl;

        ImuProcessor imu(config.gravity);
        const double init_start = dataset.imu_samples.front().t;
        if (!imu.initialize(dataset.imu_samples, init_start, config.imu_init_duration_sec)) {
            result.status = VioStatus::Failed;
            result.error = "IMU initialization failed";
            return result;
        }
        ImuPreintegrator preintegrator(config.gravity);

        VisualFrontendParams frontend_params;
        frontend_params.tracker.winSize = config.tracker_win_size;
        frontend_params.tracker.maxLevel = config.tracker_max_level;
        frontend_params.tracker.maxIters = config.tracker_max_iters;
        frontend_params.tracker.eps = config.tracker_eps;

        VisualFrontend frontend;
        frontend.setParams(frontend_params);

        VioPipeline pipeline(
            runtime_intrinsics,
            VioPipelineParams{},
            config.triangulation,
            PnPParams{},
            extrinsics,
            config.gravity,
            config.estimator
        );

        pipeline.setImuProcessor(&imu);

        StreamerConfig streamer_config;
        streamer_config.realtime = config.stream_realtime;
        streamer_config.rate = config.stream_rate;
        streamer_config.max_image_queue = config.stream_max_image_queue;

        DatasetStreamer streamer(dataset, streamer_config);
        streamer.start();

        bool have_frame = false;
        std::optional<double> previous_camera_timestamp;
        int loop_count = 0;
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

            if (camera.timestamp_s < imu.initEndTime()) {
                continue;
            }
            
            loop_count++;

            drainStreamedImuSamples(
                streamer.imuQueue(),
                imu_lookahead,
                camera.timestamp_s
            );
            imu.propagateUntil(camera.timestamp_s);
            const Pose imu_pose = imu.getCurrentPose();
            const FrameState state = frameStateFromBodyPose(
                static_cast<int>(camera.frame_index),
                camera.timestamp_s,
                imu_pose,
                extrinsics
            );

            std::optional<PreintegratedImuMeasurement> imu_factor;
            if (previous_camera_timestamp.has_value()) {
                imu_factor = preintegrator.integrate(
                    dataset.imu_samples,
                    *previous_camera_timestamp,
                    camera.timestamp_s,
                    imu.gyroBias(),
                    imu.accelBias()
                );
            }

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

            if (loop_count % 100 == 0) {
                std::cout << "[Loop " << loop_count << "] "
                          << "Tracks: " << output.tracks.size() << ", "
                          << "enough: " << (output.enough_tracks ? "YES" : "no") << ", "
                          << "Status: " << static_cast<int>(pipeline.status()) << ", "
                          << "Landmarks: " << pipeline.landmarks().size() << ", "
                          << "Pose: " << state.t_wc.transpose() << std::endl;
            }

            pipeline.processFrame(output.frame, imu_factor);

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
            previous_camera_timestamp = camera.timestamp_s;

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

bool VioPipeline::hasMetricEstimatorState() const {
    return estimator_status_ == SlidingWindowEstimatorStatus::Initialized ||
           estimator_status_ == SlidingWindowEstimatorStatus::Optimized;
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

    // Only use pose baseline if we already have landmarks and PnP is working.
    // Otherwise, the flying IMU will trigger triangulation with huge baselines that fail.
    const bool pose_ready = !landmark_map_.empty() && poseBaselineReady(current);
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

bool VioPipeline::makeTwoViewInlierPair(
    TrackedFrame& pivot_inliers,
    TrackedFrame& current_inliers
) const {
    if (!pivot_valid_) {
        return false;
    }

    std::unordered_map<int, Eigen::Vector2d> pivot_uv_by_track;
    pivot_uv_by_track.reserve(pivot_.observations.size());

    for (const Observation& obs : pivot_.observations) {
        if (obs.valid && obs.track_id >= 0 && obs.uv.allFinite()) {
            pivot_uv_by_track[obs.track_id] = obs.uv;
        }
    }

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    std::vector<int> track_ids;
    points1.reserve(current_inliers.observations.size());
    points2.reserve(current_inliers.observations.size());
    track_ids.reserve(current_inliers.observations.size());

    for (const Observation& obs : current_inliers.observations) {
        if (!obs.valid || obs.track_id < 0 || !obs.uv.allFinite()) {
            continue;
        }

        const auto it = pivot_uv_by_track.find(obs.track_id);
        if (it == pivot_uv_by_track.end()) {
            continue;
        }

        points1.emplace_back(
            static_cast<float>(it->second.x()),
            static_cast<float>(it->second.y())
        );
        points2.emplace_back(
            static_cast<float>(obs.uv.x()),
            static_cast<float>(obs.uv.y())
        );
        track_ids.push_back(obs.track_id);
    }

    if (static_cast<int>(points1.size()) < params_.min_epipolar_inliers) {
        return false;
    }

    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        intrinsics_.fx, 0.0, intrinsics_.cx,
        0.0, intrinsics_.fy, intrinsics_.cy,
        0.0, 0.0, 1.0
    );

    cv::Mat inlier_mask;
    const cv::Mat essential = cv::findEssentialMat(
        points1,
        points2,
        camera_matrix,
        cv::RANSAC,
        params_.epipolar_confidence,
        params_.epipolar_ransac_threshold_px,
        inlier_mask
    );

    if (essential.empty() || inlier_mask.empty()) {
        return false;
    }

    std::unordered_set<int> inlier_tracks;
    inlier_tracks.reserve(track_ids.size());
    const std::size_t mask_count = std::min(inlier_mask.total(), track_ids.size());
    for (std::size_t i = 0; i < mask_count; ++i) {
        if (inlier_mask.at<unsigned char>(static_cast<int>(i)) != 0) {
            inlier_tracks.insert(track_ids[i]);
        }
    }

    if (static_cast<int>(inlier_tracks.size()) < params_.min_epipolar_inliers) {
        return false;
    }

    for (Observation& obs : pivot_inliers.observations) {
        if (obs.track_id < 0 || inlier_tracks.find(obs.track_id) == inlier_tracks.end()) {
            obs.valid = false;
        }
    }
    for (Observation& obs : current_inliers.observations) {
        if (obs.track_id < 0 || inlier_tracks.find(obs.track_id) == inlier_tracks.end()) {
            obs.valid = false;
        }
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

void VioPipeline::seedMetricMotionPrior(TrackedFrame& current) const {
    if (!hasMetricEstimatorState() || frames_.empty() || landmark_map_.empty()) {
        return;
    }

    const FrameState& previous = frames_.back().state;
    const double dt = current.state.timestamp - previous.timestamp;
    if (dt <= 1e-6 || !std::isfinite(dt)) {
        return;
    }

    Eigen::Vector3d velocity = previous.v_w;
    if (!velocity.allFinite() || velocity.norm() > 3.0) {
        velocity.setZero();
    }

    current.state.q_wc = previous.q_wc;
    current.state.t_wc = previous.t_wc + velocity * dt;
    current.state.v_w = velocity;
    current.state.a_w.setZero();
}

bool VioPipeline::triangulateFromPivot(const TrackedFrame& current, int min_created) {
    if (!pivot_valid_) {
        return false;
    }

    TrackedFrame pivot_inliers = pivot_;
    TrackedFrame current_inliers = current;
    if (!makeTwoViewInlierPair(pivot_inliers, current_inliers)) {
        return false;
    }

    std::vector<TrackedFrame> pair;
    pair.reserve(2);
    pair.push_back(std::move(pivot_inliers));
    pair.push_back(std::move(current_inliers));

    const std::vector<Landmark> new_landmarks =
        triangulateLandmarks(pair, intrinsics_, triangulation_params_);

    int created = 0;
    for (const Landmark& landmark : new_landmarks) {
        if (!landmark.valid) {
            continue;
        }
        const bool is_new_track = !landmark_map_.hasTrack(landmark.track_id);
        landmark_map_.addOrUpdate(
            landmark.track_id,
            landmark.p_w,
            landmark.reprojection_error,
            landmark.num_observations
        );
        if (is_new_track) {
            ++created;
        }
    }

    return created >= min_created && created > 0;
}

bool VioPipeline::refinePoseWithPnP(TrackedFrame& current) {
    if (!pivot_valid_) {
        return false;
    }

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

    const FrameState& reference =
        frames_.empty() ? current.state : frames_.back().state;
    const double jump = (pnp_result.pose.t_wc - reference.t_wc).norm();

    const double max_step = hasMetricEstimatorState()
        ? params_.max_pose_baseline
        : maxReasonableFrameStep(reference.timestamp, current.state.timestamp);
    if (!std::isfinite(jump) || jump > max_step) {
        return false;
    }

    current.state = pnp_result.pose;
    updateVelocityFromPrevious(current.state);
    return true;
}

void VioPipeline::updateVelocityFromPrevious(FrameState& state) const {
    if (frames_.empty()) {
        return;
    }

    const FrameState& previous = frames_.back().state;
    const double dt = state.timestamp - previous.timestamp;
    if (dt <= 1e-6 || !std::isfinite(dt)) {
        return;
    }

    const Eigen::Vector3d velocity = (state.t_wc - previous.t_wc) / dt;
    if (!velocity.allFinite() || velocity.norm() > 3.0) {
        state.v_w.setZero();
        state.a_w.setZero();
        return;
    }

    state.v_w = velocity;
    state.a_w.setZero();
}

NavState VioPipeline::navStateFromFrameState(const FrameState& frame) const {
    NavState state;
    state.timestamp = frame.timestamp;

    const Eigen::Matrix3d R_wc = frame.q_wc.normalized().toRotationMatrix();
    const Eigen::Matrix3d R_wb = R_wc * extrinsics_.R_BC().transpose();
    state.q_wb = Eigen::Quaterniond(R_wb).normalized();
    state.p_wb = frame.t_wc - R_wb * extrinsics_.t_BC();
    state.v_wb = frame.v_w;
    state.a_wb = frame.a_w;

    if (imu_processor_ != nullptr) {
        state.gyro_bias = imu_processor_->gyroBias();
        state.accel_bias = imu_processor_->accelBias();
    }

    return state;
}

void VioPipeline::finalizeFrame(
    TrackedFrame& current,
    const std::optional<PreintegratedImuMeasurement>& imu_preintegration
) {
    const NavState predicted_state = navStateFromFrameState(current.state);
    const SlidingWindowEstimatorResult estimate = estimator_.addFrame(
        current,
        imu_preintegration,
        landmark_map_.getValidLandmarks(),
        predicted_state
    );
    estimator_status_ = estimate.status;

    const FrameState& reference =
        frames_.empty() ? current.state : frames_.back().state;
    const double estimate_step =
        (estimate.camera_pose.t_wc - reference.t_wc).norm();
    const double max_estimate_step =
        hasMetricEstimatorState()
            ? params_.max_pose_baseline
            : maxReasonableFrameStep(reference.timestamp, current.state.timestamp);

    const bool estimate_is_reasonable =
        estimate.success &&
        isFiniteFrameState(estimate.camera_pose) &&
        estimate.state.q_wb.coeffs().allFinite() &&
        estimate.state.p_wb.allFinite() &&
        estimate.state.v_wb.allFinite() &&
        estimate.state.a_wb.allFinite() &&
        std::isfinite(estimate_step) &&
        (estimate.metric_initialized || estimate_step <= max_estimate_step);

    if (estimate.success && !estimate_is_reasonable) {
        estimator_status_ = SlidingWindowEstimatorStatus::Degraded;
    }

    if (estimate_is_reasonable) {
        current.state = estimate.camera_pose;
        updateVelocityFromPrevious(current.state);
        for (const Landmark& landmark : estimate.landmarks) {
            if (!landmark.valid) {
                continue;
            }
            landmark_map_.addOrUpdate(
                landmark.track_id,
                landmark.p_w,
                landmark.reprojection_error,
                landmark.num_observations
            );
        }

        if (estimate.metric_initialized) {
            setPivot(current, landmark_map_.empty()
                ? Stage::NeedInitialLandmarks
                : Stage::TrackWithLandmarks);
        } else if (pivot_valid_ && pivot_.state.frame_id == current.state.frame_id) {
            pivot_ = current;
        }

        if (imu_processor_ != nullptr) {
            NavState corrected_state = navStateFromFrameState(current.state);
            corrected_state.gyro_bias = estimate.state.gyro_bias;
            corrected_state.accel_bias = estimate.state.accel_bias;
            imu_processor_->resetStateFromNavState(corrected_state);
        }
    }

    frames_.push_back(current);
}

void VioPipeline::resetTrackingSegment(const TrackedFrame& current) {
    setPivot(current, landmark_map_.empty()
        ? Stage::NeedInitialLandmarks
        : Stage::TrackWithLandmarks);
}

} // namespace vio
