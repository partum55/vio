#pragma once

#include "core/types.hpp"
#include "estimation/sliding_window_estimator.hpp"
#include "geometry/landmark_map.hpp"
#include "geometry/pnp_solver.hpp"
#include "geometry/triangulator.hpp"
#include "imu/imu_preintegration.hpp"
#include "imu/imu_processor.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vio {

    enum class VioStatus {
        Uninitialized,
        NeedFirstFrame,
        TrackingFromPivot,
        NeedInitialLandmarks,
        TrackingWithMap,
        LostTracking,
        Finished,
        Failed
    };

    struct VioPipelineParams {
        int min_tracked_points = 15;
        int min_landmarks_for_pnp = 10;
        int min_landmarks_after_initial_triangulation = 5;

        // Visual-only fallback: use robust 2D parallax between the pivot and
        // current frame when the IMU pose prior is unavailable/weak.
        double min_pixel_baseline = 1.0;
        double max_pixel_baseline = 60.0;
        int min_shared_tracks_for_baseline = 10;

        // Reject obviously broken pose jumps before triangulation/PnP refresh.
        double max_pose_baseline = 1.2;
        double min_pose_baseline_translation = 0.05;
        double min_pose_baseline_rotation_deg = 3.0;

        // PnP must have enough RANSAC inliers to be allowed to update the pose.
        int min_pnp_inliers = 14;

        // Two-view RANSAC gate before triangulation.
        int min_epipolar_inliers = 15;
        double epipolar_ransac_threshold_px = 2.0;
        double epipolar_confidence = 0.999;
    };

    struct VioRunConfig {
        std::string imu_csv_path;
        std::string images_dir;
        std::string frame_timestamps_path;

        std::string output_poses_csv = "poses.csv";
        std::string output_observations_csv = "observations.csv";
        std::string output_video_path = "imu_tracking_visualization.mp4";
        std::string output_landmarks_csv = "landmarks.csv";

        Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, 9.81);
        CameraIntrinsics camera_intrinsics;
        Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();
        bool use_dataset_calibration = true;
        bool images_are_undistorted = true;
        double imu_init_duration_sec = 3.0;
        TriangulationParams triangulation;

        int tracker_win_size = 9;
        int tracker_max_level = 3;
        int tracker_max_iters = 10;
        float tracker_eps = 1e-3f;

        bool stream_realtime = false;
        double stream_rate = 1.0;
        std::size_t stream_max_image_queue = 8;
        SlidingWindowEstimatorParams estimator;

        std::function<void(
            const TrackedFrame& frame,
            const std::vector<Track>& tracks,
            const std::vector<Landmark>& landmarks,
            VioStatus status,
            bool pose_reliable,
            const std::filesystem::path& image_path)> frame_logger;
    };

    struct VioRunResult {
        bool success = false;
        VioStatus status = VioStatus::Uninitialized;
        std::size_t frame_count = 0;
        std::size_t landmark_count = 0;
        std::string error;
    };

    class VioPipeline {
    public:
        explicit VioPipeline(
            const CameraIntrinsics& intrinsics,
            const VioPipelineParams& params = VioPipelineParams{},
            const TriangulationParams& triangulation_params = TriangulationParams{},
            const PnPParams& pnp_params = PnPParams{},
            const CameraImuExtrinsics& extrinsics = CameraImuExtrinsics{},
            const Eigen::Vector3d& gravity = Eigen::Vector3d(0.0, 0.0, 9.81),
            const SlidingWindowEstimatorParams& estimator_params =
                SlidingWindowEstimatorParams{}
        );

        void reset();
        void processFrame(const TrackedFrame& input_frame);
        void processFrame(
            const TrackedFrame& input_frame,
            const std::optional<PreintegratedImuMeasurement>& imu_preintegration
        );

        const std::vector<TrackedFrame>& frames() const;
        const LandmarkMap& landmarks() const;
        VioStatus status() const;
        bool hasCurrentPose() const;
        FrameState currentPose() const;
        SlidingWindowEstimatorStatus estimatorStatus() const;

        static VioRunResult runConfigured(const VioRunConfig& config);

        void setImuProcessor(ImuProcessor* imu_processor);

    private:
        enum class Stage {
            NeedPivot,
            NeedInitialLandmarks,
            TrackWithLandmarks
        };

        int validObservationCount(const TrackedFrame& frame) const;
        int pnpCorrespondenceCount(const TrackedFrame& frame) const;
        bool hasMetricEstimatorState() const;

        double poseBaseline(const TrackedFrame& current) const;
        double robustPixelBaseline(const TrackedFrame& current, int* shared_count = nullptr) const;
        bool poseBaselineReady(const TrackedFrame& current) const;
        bool baselineReady(const TrackedFrame& current) const;
        bool baselineReasonable(const TrackedFrame& current) const;
        bool makeTwoViewInlierPair(
            TrackedFrame& pivot_inliers,
            TrackedFrame& current_inliers
        ) const;

        void setPivot(const TrackedFrame& frame, Stage next_stage);
        void initializePivot(const TrackedFrame& frame);
        void seedMetricMotionPrior(TrackedFrame& current) const;
        bool triangulateFromPivot(const TrackedFrame& current, int min_created);
        bool refinePoseWithPnP(TrackedFrame& current);
        void updateVelocityFromPrevious(FrameState& state) const;
        NavState navStateFromFrameState(const FrameState& frame) const;
        void finalizeFrame(
            TrackedFrame& current,
            const std::optional<PreintegratedImuMeasurement>& imu_preintegration
        );
        void resetTrackingSegment(const TrackedFrame& current);

    private:
        CameraIntrinsics intrinsics_;
        VioPipelineParams params_;
        TriangulationParams triangulation_params_;
        PnPSolver pnp_solver_;
        CameraImuExtrinsics extrinsics_;
        SlidingWindowEstimator estimator_;
        SlidingWindowEstimatorStatus estimator_status_ =
            SlidingWindowEstimatorStatus::Uninitialized;

        Stage stage_ = Stage::NeedPivot;
        VioStatus status_ = VioStatus::Uninitialized;
        TrackedFrame pivot_;
        bool pivot_valid_ = false;
        LandmarkMap landmark_map_;

        std::vector<TrackedFrame> frames_;
        std::vector<PreintegratedImuMeasurement> imu_preintegrations_;

        ImuProcessor* imu_processor_ = nullptr;
    };

} // namespace vio
