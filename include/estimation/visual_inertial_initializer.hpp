#pragma once

#include "estimation/sliding_window_estimator.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace vio {

struct VisualInertialInitializationResult {
    bool success = false;
    double scale = 1.0;
    double rmse = 0.0;
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    std::vector<SlidingWindowEstimator::WindowState> states;
    std::vector<TrackedFrame> frames;
    std::vector<PreintegratedImuMeasurement> imu_factors;
    std::unordered_map<int, SlidingWindowEstimator::LandmarkState> landmarks;
    std::string failure_reason;
};

class VisualInertialInitializer {
public:
    [[nodiscard]] static VisualInertialInitializationResult initialize(
        const std::vector<SlidingWindowEstimator::WindowState>& states,
        const std::vector<TrackedFrame>& frames,
        const std::vector<PreintegratedImuMeasurement>& imu_factors,
        const std::unordered_map<int, SlidingWindowEstimator::LandmarkState>& landmarks,
        const CameraImuExtrinsics& extrinsics,
        const Eigen::Vector3d& gravity,
        const SlidingWindowEstimatorParams& params
    );
};

} // namespace vio
