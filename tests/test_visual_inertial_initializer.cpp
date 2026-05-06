#include "estimation/visual_inertial_initializer.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {

using vio::PreintegratedImuMeasurement;
using vio::SlidingWindowEstimator;
using vio::SlidingWindowEstimatorParams;
using vio::TrackedFrame;

SlidingWindowEstimator::WindowState makeState(
    int frame_id,
    double timestamp,
    const Eigen::Vector3d& p_visual
) {
    SlidingWindowEstimator::WindowState state;
    state.frame_id = frame_id;
    state.timestamp = timestamp;
    state.q_wb[0] = 1.0;
    state.p_wb[0] = p_visual.x();
    state.p_wb[1] = p_visual.y();
    state.p_wb[2] = p_visual.z();
    return state;
}

TrackedFrame makeFrame(
    int frame_id,
    double timestamp,
    const Eigen::Vector3d& p_visual
) {
    TrackedFrame frame;
    frame.state.frame_id = frame_id;
    frame.state.timestamp = timestamp;
    frame.state.q_wc = Eigen::Quaterniond::Identity();
    frame.state.t_wc = p_visual;
    return frame;
}

PreintegratedImuMeasurement makeFactor(
    double start,
    double end,
    const Eigen::Vector3d& p_i,
    const Eigen::Vector3d& p_j,
    const Eigen::Vector3d& v_i,
    const Eigen::Vector3d& v_j,
    const Eigen::Vector3d& gravity
) {
    const double dt = end - start;
    PreintegratedImuMeasurement factor;
    factor.start_time = start;
    factor.end_time = end;
    factor.dt = dt;
    factor.delta_q = Eigen::Quaterniond::Identity();
    factor.delta_p = p_j - p_i - v_i * dt + 0.5 * gravity * dt * dt;
    factor.delta_v = v_j - v_i - gravity * dt;
    factor.jacobian_q_bias_gyro = -Eigen::Matrix3d::Identity() * dt;
    factor.jacobian_p_bias_accel =
        -0.5 * Eigen::Matrix3d::Identity() * dt * dt;
    factor.jacobian_v_bias_accel = -Eigen::Matrix3d::Identity() * dt;
    factor.valid = true;
    factor.num_samples = 1;
    return factor;
}

} // namespace

int main()
{
    constexpr double kScale = 0.2;
    const Eigen::Vector3d gravity(0.0, 0.0, 9.80665);

    std::vector<Eigen::Vector3d> p_metric = {
        {0.0, 0.0, 0.0},
        {0.8, 0.1, 0.0},
        {1.7, 0.3, 0.1},
        {2.9, 0.8, 0.2},
        {4.2, 1.2, 0.2},
    };
    std::vector<Eigen::Vector3d> v_metric = {
        {0.8, 0.1, 0.0},
        {0.9, 0.2, 0.1},
        {1.2, 0.5, 0.1},
        {1.3, 0.4, 0.0},
        {1.3, 0.4, 0.0},
    };

    std::vector<SlidingWindowEstimator::WindowState> states;
    std::vector<TrackedFrame> frames;
    std::vector<PreintegratedImuMeasurement> factors;
    for (std::size_t i = 0; i < p_metric.size(); ++i) {
        const Eigen::Vector3d p_visual = p_metric[i] / kScale;
        states.push_back(makeState(static_cast<int>(i), static_cast<double>(i), p_visual));
        frames.push_back(makeFrame(static_cast<int>(i), static_cast<double>(i), p_visual));
        if (i + 1 < p_metric.size()) {
            factors.push_back(makeFactor(
                static_cast<double>(i),
                static_cast<double>(i + 1),
                p_metric[i],
                p_metric[i + 1],
                v_metric[i],
                v_metric[i + 1],
                gravity
            ));
        }
    }

    std::unordered_map<int, SlidingWindowEstimator::LandmarkState> landmarks;
    SlidingWindowEstimator::LandmarkState landmark;
    landmark.track_id = 1;
    landmark.p_w[0] = 5.0 / kScale;
    landmark.p_w[1] = 1.0 / kScale;
    landmark.p_w[2] = 3.0 / kScale;
    landmark.valid = true;
    landmarks[1] = landmark;

    SlidingWindowEstimatorParams params;
    params.init_min_frames = static_cast<int>(states.size());
    params.init_max_frames = static_cast<int>(states.size());
    params.init_max_alignment_rmse = 1e-6;
    params.init_gravity_norm_tolerance = 1e-6;

    const vio::VisualInertialInitializationResult result =
        vio::VisualInertialInitializer::initialize(
            states,
            frames,
            factors,
            landmarks,
            vio::CameraImuExtrinsics{},
            gravity,
            params
        );

    if (!result.success) {
        std::cerr << "initializer failed: " << result.failure_reason << "\n";
        return 1;
    }
    if (std::abs(result.scale - kScale) > 1e-6) {
        std::cerr << "scale mismatch: " << result.scale << "\n";
        return 1;
    }
    if ((result.frames.back().state.t_wc - p_metric.back()).norm() > 1e-6) {
        std::cerr << "scaled final pose mismatch\n";
        return 1;
    }
    return 0;
}
