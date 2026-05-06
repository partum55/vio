#pragma once

#include "core/types.hpp"
#include "imu/imu_preintegration.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vio {

enum class SlidingWindowEstimatorStatus {
    Uninitialized,
    Initializing,
    Initialized,
    Optimized,
    Degraded,
    Failed
};

struct SlidingWindowEstimatorParams {
    std::size_t max_window_size = 10;
    int min_visual_observations = 15;
    int min_landmarks = 8;
    int min_initial_frames = 2;
    int min_initial_imu_factors = 1;
    double min_initial_parallax_px = 2.0;
    double init_window_time_sec = 5.0;
    int init_min_frames = 80;
    int init_max_frames = 160;
    double init_min_parallax_px = 20.0;
    double init_gravity_norm_tolerance = 1.0;
    double init_min_scale = 1e-4;
    double init_max_scale = 1e4;
    double init_min_reciprocal_condition = 1e-8;
    double init_max_alignment_rmse = 0.5;
    double init_max_accel_bias_norm = 2.5;

    int max_solver_iterations = 12;
    double reprojection_sigma_px = 2.0;
    double max_reprojection_error_px = 8.0;
    double imu_position_sigma = 0.08;
    double imu_velocity_sigma = 0.15;
    double imu_rotation_sigma = 0.03;
    double bias_random_walk_sigma = 0.02;
    double prior_position_sigma = 0.05;
    double prior_velocity_sigma = 0.10;
    double prior_rotation_sigma = 0.02;
    double bias_repropagation_threshold = 0.01;
};

struct SlidingWindowEstimatorResult {
    bool success = false;
    SlidingWindowEstimatorStatus status = SlidingWindowEstimatorStatus::Uninitialized;
    NavState state;
    FrameState camera_pose;
    std::vector<Landmark> landmarks;
    int visual_residuals = 0;
    int rejected_observations = 0;
    double final_cost = 0.0;
    bool metric_initialized = false;
    double estimated_scale = 1.0;
    double initialization_rmse = 0.0;
};

class SlidingWindowEstimator {
public:
    explicit SlidingWindowEstimator(
        const CameraIntrinsics& intrinsics,
        const CameraImuExtrinsics& extrinsics,
        const SlidingWindowEstimatorParams& params = SlidingWindowEstimatorParams{},
        const Eigen::Vector3d& gravity = Eigen::Vector3d(0.0, 0.0, 9.81)
    );

    void reset();

    [[nodiscard]] SlidingWindowEstimatorResult addFrame(
        const TrackedFrame& frame,
        const std::optional<PreintegratedImuMeasurement>& imu_factor,
        const std::vector<Landmark>& landmarks,
        const NavState& predicted_state
    );

public:
    struct WindowState {
        int frame_id = -1;
        double timestamp = 0.0;
        double q_wb[4] = {1.0, 0.0, 0.0, 0.0};
        double p_wb[3] = {0.0, 0.0, 0.0};
        double v_wb[3] = {0.0, 0.0, 0.0};
        double a_wb[3] = {0.0, 0.0, 0.0};
        double gyro_bias[3] = {0.0, 0.0, 0.0};
        double accel_bias[3] = {0.0, 0.0, 0.0};
    };

    struct LandmarkState {
        int track_id = -1;
        double p_w[3] = {0.0, 0.0, 0.0};
        bool valid = false;
    };

private:
    [[nodiscard]] NavState navStateFromWindowState(
        const WindowState& state
    ) const;

    void seedLandmarks(const std::vector<Landmark>& landmarks);
    void dropOldWindowItems();
    void dropOldInitializationItems();
    [[nodiscard]] int countVisualResiduals() const;
    [[nodiscard]] double medianParallaxPx() const;
    [[nodiscard]] bool hasUsableInitialization() const;
    [[nodiscard]] bool hasMetricInitializationWindow() const;
    void applyInitializedLandmarksToResult(
        SlidingWindowEstimatorResult& result
    ) const;
    int rejectReprojectionOutliers();

    CameraIntrinsics intrinsics_;
    CameraImuExtrinsics extrinsics_;
    SlidingWindowEstimatorParams params_;
    Eigen::Vector3d gravity_;
    SlidingWindowEstimatorStatus status_ = SlidingWindowEstimatorStatus::Uninitialized;
    std::vector<WindowState> states_;
    std::vector<TrackedFrame> window_;
    std::vector<PreintegratedImuMeasurement> imu_factors_;
    std::unordered_map<int, LandmarkState> landmarks_;
    bool metric_initialized_ = false;
};

FrameState cameraFrameFromNavState(
    int frame_id,
    const NavState& state,
    const CameraImuExtrinsics& extrinsics
);

} // namespace vio
