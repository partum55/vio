#include "estimation/sliding_window_estimator.hpp"

#include "estimation/visual_inertial_initializer.hpp"

#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vio {
namespace {

template <typename T>
void rotateByInverseQuaternion(const T* const q, const T* const point, T* rotated)
{
    const T q_inv[4] = {q[0], -q[1], -q[2], -q[3]};
    ceres::QuaternionRotatePoint(q_inv, point, rotated);
}

struct ReprojectionResidual {
    ReprojectionResidual(
        const CameraIntrinsics& intrinsics,
        const CameraImuExtrinsics& extrinsics,
        const Eigen::Vector2d& uv,
        double sigma
    )
        : fx(intrinsics.fx),
          fy(intrinsics.fy),
          cx(intrinsics.cx),
          cy(intrinsics.cy),
          u(uv.x()),
          v(uv.y()),
          inv_sigma(1.0 / std::max(sigma, 1e-6))
    {
        const Eigen::Matrix3d R_CB = extrinsics.R_BC().transpose();
        const Eigen::Vector3d t_BC = extrinsics.t_BC();
        for (int r = 0; r < 3; ++r) {
            t_bc[r] = t_BC(r);
            for (int c = 0; c < 3; ++c) {
                r_cb[r * 3 + c] = R_CB(r, c);
            }
        }
    }

    template <typename T>
    bool operator()(
        const T* const q_wb,
        const T* const p_wb,
        const T* const landmark_w,
        T* residuals
    ) const {
        T pw_minus_p[3] = {
            landmark_w[0] - p_wb[0],
            landmark_w[1] - p_wb[1],
            landmark_w[2] - p_wb[2]
        };

        T p_b[3];
        rotateByInverseQuaternion(q_wb, pw_minus_p, p_b);

        const T p_b_cam[3] = {
            p_b[0] - T(t_bc[0]),
            p_b[1] - T(t_bc[1]),
            p_b[2] - T(t_bc[2])
        };

        T p_c[3];
        for (int r = 0; r < 3; ++r) {
            p_c[r] = T(r_cb[r * 3 + 0]) * p_b_cam[0] +
                     T(r_cb[r * 3 + 1]) * p_b_cam[1] +
                     T(r_cb[r * 3 + 2]) * p_b_cam[2];
        }

        const T z = p_c[2] + T(1e-9);
        const T u_hat = T(fx) * p_c[0] / z + T(cx);
        const T v_hat = T(fy) * p_c[1] / z + T(cy);

        residuals[0] = (u_hat - T(u)) * T(inv_sigma);
        residuals[1] = (v_hat - T(v)) * T(inv_sigma);
        return true;
    }

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double u = 0.0;
    double v = 0.0;
    double inv_sigma = 1.0;
    double r_cb[9] = {};
    double t_bc[3] = {};
};

struct ImuResidual {
    ImuResidual(
        const PreintegratedImuMeasurement& factor,
        const Eigen::Vector3d& gravity,
        const SlidingWindowEstimatorParams& params
    )
        : dt(factor.dt),
          gravity_w{gravity.x(), gravity.y(), gravity.z()},
          delta_q{factor.delta_q.w(), factor.delta_q.x(), factor.delta_q.y(), factor.delta_q.z()},
          delta_p{factor.delta_p.x(), factor.delta_p.y(), factor.delta_p.z()},
          delta_v{factor.delta_v.x(), factor.delta_v.y(), factor.delta_v.z()},
          bg0{factor.linearization_gyro_bias.x(), factor.linearization_gyro_bias.y(), factor.linearization_gyro_bias.z()},
          ba0{factor.linearization_accel_bias.x(), factor.linearization_accel_bias.y(), factor.linearization_accel_bias.z()},
          inv_pos_sigma(1.0 / std::max(
              std::sqrt(std::max(factor.covariance(0, 0), 0.0)),
              params.imu_position_sigma)),
          inv_vel_sigma(1.0 / std::max(
              std::sqrt(std::max(factor.covariance(3, 3), 0.0)),
              params.imu_velocity_sigma)),
          inv_rot_sigma(1.0 / std::max(
              std::sqrt(std::max(factor.covariance(6, 6), 0.0)),
              params.imu_rotation_sigma)),
          inv_bias_sigma(1.0 / std::max(params.bias_random_walk_sigma * std::sqrt(std::max(factor.dt, 1e-3)), 1e-6))
    {
        const Eigen::Quaterniond dq_inv = factor.delta_q.conjugate().normalized();
        delta_q_inv[0] = dq_inv.w();
        delta_q_inv[1] = dq_inv.x();
        delta_q_inv[2] = dq_inv.y();
        delta_q_inv[3] = dq_inv.z();

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                jpba[r * 3 + c] = factor.jacobian_p_bias_accel(r, c);
                jpbg[r * 3 + c] = factor.jacobian_p_bias_gyro(r, c);
                jvba[r * 3 + c] = factor.jacobian_v_bias_accel(r, c);
                jvbg[r * 3 + c] = factor.jacobian_v_bias_gyro(r, c);
                jqbg[r * 3 + c] = factor.jacobian_q_bias_gyro(r, c);
            }
        }
    }

    template <typename T>
    bool operator()(
        const T* const q_i,
        const T* const p_i,
        const T* const v_i,
        const T* const bg_i,
        const T* const ba_i,
        const T* const q_j,
        const T* const p_j,
        const T* const v_j,
        const T* const bg_j,
        const T* const ba_j,
        T* residuals
    ) const {
        T corrected_delta_p[3];
        T corrected_delta_v[3];
        T dbg[3] = {
            bg_i[0] - T(bg0[0]),
            bg_i[1] - T(bg0[1]),
            bg_i[2] - T(bg0[2])
        };
        T dba[3] = {
            ba_i[0] - T(ba0[0]),
            ba_i[1] - T(ba0[1]),
            ba_i[2] - T(ba0[2])
        };
        for (int r = 0; r < 3; ++r) {
            corrected_delta_p[r] = T(delta_p[r]);
            corrected_delta_v[r] = T(delta_v[r]);
            for (int c = 0; c < 3; ++c) {
                corrected_delta_p[r] += T(jpba[r * 3 + c]) * dba[c];
                corrected_delta_p[r] += T(jpbg[r * 3 + c]) * dbg[c];
                corrected_delta_v[r] += T(jvba[r * 3 + c]) * dba[c];
                corrected_delta_v[r] += T(jvbg[r * 3 + c]) * dbg[c];
            }
        }

        T predicted_p_w[3] = {
            p_j[0] - p_i[0] - v_i[0] * T(dt) + T(0.5 * dt * dt * gravity_w[0]),
            p_j[1] - p_i[1] - v_i[1] * T(dt) + T(0.5 * dt * dt * gravity_w[1]),
            p_j[2] - p_i[2] - v_i[2] * T(dt) + T(0.5 * dt * dt * gravity_w[2])
        };
        T predicted_p_i[3];
        rotateByInverseQuaternion(q_i, predicted_p_w, predicted_p_i);

        T predicted_v_w[3] = {
            v_j[0] - v_i[0] - T(dt * gravity_w[0]),
            v_j[1] - v_i[1] - T(dt * gravity_w[1]),
            v_j[2] - v_i[2] - T(dt * gravity_w[2])
        };
        T predicted_v_i[3];
        rotateByInverseQuaternion(q_i, predicted_v_w, predicted_v_i);

        for (int k = 0; k < 3; ++k) {
            residuals[k] = (predicted_p_i[k] - corrected_delta_p[k]) * T(inv_pos_sigma);
            residuals[3 + k] = (predicted_v_i[k] - corrected_delta_v[k]) * T(inv_vel_sigma);
        }

        T q_i_inv[4] = {q_i[0], -q_i[1], -q_i[2], -q_i[3]};
        T q_ij[4];
        ceres::QuaternionProduct(q_i_inv, q_j, q_ij);
        T dq_res[4];
        const T delta_q_inv_t[4] = {
            T(delta_q_inv[0]), T(delta_q_inv[1]), T(delta_q_inv[2]), T(delta_q_inv[3])
        };
        ceres::QuaternionProduct(delta_q_inv_t, q_ij, dq_res);

        for (int r = 0; r < 3; ++r) {
            T bias_rot = T(0.0);
            for (int c = 0; c < 3; ++c) {
                bias_rot += T(jqbg[r * 3 + c]) * dbg[c];
            }
            residuals[6 + r] = (T(2.0) * dq_res[1 + r] - bias_rot) * T(inv_rot_sigma);
            residuals[9 + r] = (bg_j[r] - bg_i[r]) * T(inv_bias_sigma);
            residuals[12 + r] = (ba_j[r] - ba_i[r]) * T(inv_bias_sigma);
        }
        return true;
    }

    double dt = 0.0;
    double gravity_w[3] = {};
    double delta_q[4] = {};
    double delta_q_inv[4] = {};
    double delta_p[3] = {};
    double delta_v[3] = {};
    double bg0[3] = {};
    double ba0[3] = {};
    double jpba[9] = {};
    double jpbg[9] = {};
    double jvba[9] = {};
    double jvbg[9] = {};
    double jqbg[9] = {};
    double inv_pos_sigma = 1.0;
    double inv_vel_sigma = 1.0;
    double inv_rot_sigma = 1.0;
    double inv_bias_sigma = 1.0;
};

struct StatePriorResidual {
    StatePriorResidual(
        const SlidingWindowEstimator::WindowState& state,
        const SlidingWindowEstimatorParams& params
    )
        : q0{state.q_wb[0], state.q_wb[1], state.q_wb[2], state.q_wb[3]},
          p0{state.p_wb[0], state.p_wb[1], state.p_wb[2]},
          v0{state.v_wb[0], state.v_wb[1], state.v_wb[2]},
          bg0{state.gyro_bias[0], state.gyro_bias[1], state.gyro_bias[2]},
          ba0{state.accel_bias[0], state.accel_bias[1], state.accel_bias[2]},
          inv_pos_sigma(1.0 / std::max(params.prior_position_sigma, 1e-6)),
          inv_vel_sigma(1.0 / std::max(params.prior_velocity_sigma, 1e-6)),
          inv_rot_sigma(1.0 / std::max(params.prior_rotation_sigma, 1e-6)),
          inv_bias_sigma(1.0 / std::max(params.bias_random_walk_sigma, 1e-6))
    {
        const Eigen::Quaterniond q(q0[0], q0[1], q0[2], q0[3]);
        const Eigen::Quaterniond q_inv = q.conjugate().normalized();
        q0_inv[0] = q_inv.w();
        q0_inv[1] = q_inv.x();
        q0_inv[2] = q_inv.y();
        q0_inv[3] = q_inv.z();
    }

    template <typename T>
    bool operator()(
        const T* const q,
        const T* const p,
        const T* const v,
        const T* const bg,
        const T* const ba,
        T* residuals
    ) const {
        const T q0_inv_t[4] = {T(q0_inv[0]), T(q0_inv[1]), T(q0_inv[2]), T(q0_inv[3])};
        T dq[4];
        ceres::QuaternionProduct(q0_inv_t, q, dq);
        for (int k = 0; k < 3; ++k) {
            residuals[k] = (p[k] - T(p0[k])) * T(inv_pos_sigma);
            residuals[3 + k] = (v[k] - T(v0[k])) * T(inv_vel_sigma);
            residuals[6 + k] = T(2.0) * dq[1 + k] * T(inv_rot_sigma);
            residuals[9 + k] = (bg[k] - T(bg0[k])) * T(inv_bias_sigma);
            residuals[12 + k] = (ba[k] - T(ba0[k])) * T(inv_bias_sigma);
        }
        return true;
    }

    double q0[4] = {};
    double q0_inv[4] = {};
    double p0[3] = {};
    double v0[3] = {};
    double bg0[3] = {};
    double ba0[3] = {};
    double inv_pos_sigma = 1.0;
    double inv_vel_sigma = 1.0;
    double inv_rot_sigma = 1.0;
    double inv_bias_sigma = 1.0;
};

double reprojectionErrorPx(
    const CameraIntrinsics& intrinsics,
    const CameraImuExtrinsics& extrinsics,
    const SlidingWindowEstimator::WindowState& state,
    const SlidingWindowEstimator::LandmarkState& landmark,
    const Eigen::Vector2d& uv
) {
    const Eigen::Quaterniond q_wb(
        state.q_wb[0],
        state.q_wb[1],
        state.q_wb[2],
        state.q_wb[3]
    );
    const Eigen::Vector3d p_wb(state.p_wb[0], state.p_wb[1], state.p_wb[2]);
    const Eigen::Vector3d p_w(landmark.p_w[0], landmark.p_w[1], landmark.p_w[2]);

    const Eigen::Vector3d p_b = q_wb.conjugate() * (p_w - p_wb);
    const Eigen::Vector3d p_c =
        extrinsics.R_BC().transpose() * (p_b - extrinsics.t_BC());
    if (p_c.z() <= 1e-9) {
        return std::numeric_limits<double>::infinity();
    }

    const Eigen::Vector2d projected(
        intrinsics.fx * p_c.x() / p_c.z() + intrinsics.cx,
        intrinsics.fy * p_c.y() / p_c.z() + intrinsics.cy
    );
    return (projected - uv).norm();
}

} // namespace

FrameState cameraFrameFromNavState(
    int frame_id,
    const NavState& state,
    const CameraImuExtrinsics& extrinsics
) {
    FrameState frame;
    frame.frame_id = frame_id;
    frame.timestamp = state.timestamp;

    const Eigen::Matrix3d R_wb = state.q_wb.toRotationMatrix();
    const Eigen::Matrix3d R_wc = R_wb * extrinsics.R_BC();
    frame.q_wc = Eigen::Quaterniond(R_wc).normalized();
    frame.t_wc = state.p_wb + R_wb * extrinsics.t_BC();
    frame.v_w = state.v_wb;
    frame.a_w = state.a_wb;
    return frame;
}

SlidingWindowEstimator::SlidingWindowEstimator(
    const CameraIntrinsics& intrinsics,
    const CameraImuExtrinsics& extrinsics,
    const SlidingWindowEstimatorParams& params,
    const Eigen::Vector3d& gravity
)
    : intrinsics_(intrinsics),
      extrinsics_(extrinsics),
      params_(params),
      gravity_(gravity)
{
    if (!intrinsics_.isValid()) {
        throw std::invalid_argument("SlidingWindowEstimator requires valid intrinsics");
    }
    if (params_.max_window_size == 0) {
        throw std::invalid_argument("SlidingWindowEstimator window size must be positive");
    }
}

void SlidingWindowEstimator::reset()
{
    states_.clear();
    window_.clear();
    imu_factors_.clear();
    landmarks_.clear();
    metric_initialized_ = false;
    status_ = SlidingWindowEstimatorStatus::Uninitialized;
}

NavState SlidingWindowEstimator::navStateFromWindowState(
    const WindowState& state
) const {
    NavState nav;
    nav.timestamp = state.timestamp;
    nav.q_wb = Eigen::Quaterniond(
        state.q_wb[0],
        state.q_wb[1],
        state.q_wb[2],
        state.q_wb[3]
    ).normalized();
    nav.p_wb = Eigen::Vector3d(state.p_wb[0], state.p_wb[1], state.p_wb[2]);
    nav.v_wb = Eigen::Vector3d(state.v_wb[0], state.v_wb[1], state.v_wb[2]);
    nav.a_wb = Eigen::Vector3d(state.a_wb[0], state.a_wb[1], state.a_wb[2]);
    nav.gyro_bias = Eigen::Vector3d(state.gyro_bias[0], state.gyro_bias[1], state.gyro_bias[2]);
    nav.accel_bias = Eigen::Vector3d(state.accel_bias[0], state.accel_bias[1], state.accel_bias[2]);
    return nav;
}

void SlidingWindowEstimator::seedLandmarks(const std::vector<Landmark>& landmarks)
{
    for (const Landmark& landmark : landmarks) {
        if (!landmark.valid || landmark.track_id < 0 || !landmark.p_w.allFinite()) {
            continue;
        }
        LandmarkState& state = landmarks_[landmark.track_id];
        state.track_id = landmark.track_id;
        state.p_w[0] = landmark.p_w.x();
        state.p_w[1] = landmark.p_w.y();
        state.p_w[2] = landmark.p_w.z();
        state.valid = true;
    }
}

void SlidingWindowEstimator::dropOldWindowItems()
{
    while (states_.size() > params_.max_window_size) {
        states_.erase(states_.begin());
        window_.erase(window_.begin());
        if (!imu_factors_.empty()) {
            imu_factors_.erase(imu_factors_.begin());
        }
    }
}

void SlidingWindowEstimator::dropOldInitializationItems()
{
    const std::size_t max_frames =
        static_cast<std::size_t>(std::max(params_.init_max_frames, 2));
    while (states_.size() > max_frames) {
        states_.erase(states_.begin());
        window_.erase(window_.begin());
        if (!imu_factors_.empty()) {
            imu_factors_.erase(imu_factors_.begin());
        }
    }

    while (states_.size() >= 2 &&
           params_.init_window_time_sec > 0.0 &&
           states_.back().timestamp - states_.front().timestamp >
               params_.init_window_time_sec) {
        states_.erase(states_.begin());
        window_.erase(window_.begin());
        if (!imu_factors_.empty()) {
            imu_factors_.erase(imu_factors_.begin());
        }
    }
}

int SlidingWindowEstimator::countVisualResiduals() const
{
    int count = 0;
    for (const TrackedFrame& frame : window_) {
        for (const Observation& obs : frame.observations) {
            if (!obs.valid || obs.track_id < 0 || !obs.uv.allFinite()) {
                continue;
            }
            const auto it = landmarks_.find(obs.track_id);
            if (it != landmarks_.end() && it->second.valid) {
                ++count;
            }
        }
    }
    return count;
}

double SlidingWindowEstimator::medianParallaxPx() const
{
    if (window_.size() < 2) {
        return 0.0;
    }

    const TrackedFrame& first = window_.front();
    const TrackedFrame& last = window_.back();
    std::unordered_map<int, Eigen::Vector2d> first_uv;
    first_uv.reserve(first.observations.size());
    for (const Observation& obs : first.observations) {
        if (obs.valid && obs.track_id >= 0 && obs.uv.allFinite()) {
            first_uv[obs.track_id] = obs.uv;
        }
    }

    std::vector<double> parallax;
    parallax.reserve(last.observations.size());
    for (const Observation& obs : last.observations) {
        if (!obs.valid || obs.track_id < 0 || !obs.uv.allFinite()) {
            continue;
        }
        const auto it = first_uv.find(obs.track_id);
        if (it == first_uv.end()) {
            continue;
        }
        const double distance = (obs.uv - it->second).norm();
        if (std::isfinite(distance)) {
            parallax.push_back(distance);
        }
    }

    if (parallax.empty()) {
        return 0.0;
    }
    const std::size_t mid = parallax.size() / 2;
    std::nth_element(parallax.begin(), parallax.begin() + mid, parallax.end());
    return parallax[mid];
}

bool SlidingWindowEstimator::hasUsableInitialization() const
{
    const int valid_imu_factors = static_cast<int>(std::count_if(
        imu_factors_.begin(),
        imu_factors_.end(),
        [](const PreintegratedImuMeasurement& factor) {
            return factor.valid && factor.dt > 0.0;
        }
    ));

    return states_.size() >= static_cast<std::size_t>(params_.min_initial_frames) &&
           valid_imu_factors >= params_.min_initial_imu_factors &&
           countVisualResiduals() >= params_.min_visual_observations &&
           static_cast<int>(landmarks_.size()) >= params_.min_landmarks &&
           medianParallaxPx() >= params_.min_initial_parallax_px;
}

bool SlidingWindowEstimator::hasMetricInitializationWindow() const
{
    const double duration = states_.size() < 2
        ? 0.0
        : states_.back().timestamp - states_.front().timestamp;
    const int valid_imu_factors = static_cast<int>(std::count_if(
        imu_factors_.begin(),
        imu_factors_.end(),
        [](const PreintegratedImuMeasurement& factor) {
            return factor.valid && factor.dt > 0.0;
        }
    ));

    return states_.size() >= static_cast<std::size_t>(params_.init_min_frames) &&
           duration >= 0.9 * params_.init_window_time_sec &&
           valid_imu_factors >= static_cast<int>(states_.size()) - 1 &&
           countVisualResiduals() >= params_.min_visual_observations &&
           static_cast<int>(landmarks_.size()) >= params_.min_landmarks &&
           medianParallaxPx() >= params_.init_min_parallax_px;
}

void SlidingWindowEstimator::applyInitializedLandmarksToResult(
    SlidingWindowEstimatorResult& result
) const {
    std::unordered_map<int, int> observation_counts;
    for (const TrackedFrame& window_frame : window_) {
        for (const Observation& obs : window_frame.observations) {
            if (obs.valid && obs.track_id >= 0) {
                ++observation_counts[obs.track_id];
            }
        }
    }

    result.landmarks.reserve(landmarks_.size());
    int landmark_id = 0;
    for (const auto& [track_id, landmark] : landmarks_) {
        if (!landmark.valid) {
            continue;
        }
        Landmark out;
        out.id = landmark_id++;
        out.track_id = track_id;
        out.p_w = Eigen::Vector3d(landmark.p_w[0], landmark.p_w[1], landmark.p_w[2]);
        out.valid = out.p_w.allFinite();
        out.reprojection_error = 0.0;
        out.num_observations = observation_counts[track_id];
        result.landmarks.push_back(out);
    }
}

int SlidingWindowEstimator::rejectReprojectionOutliers()
{
    int rejected = 0;
    std::unordered_map<int, int> support;

    for (std::size_t frame_idx = 0; frame_idx < window_.size(); ++frame_idx) {
        TrackedFrame& frame = window_[frame_idx];
        const WindowState& state = states_[frame_idx];
        for (Observation& obs : frame.observations) {
            if (!obs.valid || obs.track_id < 0 || !obs.uv.allFinite()) {
                continue;
            }
            auto landmark_it = landmarks_.find(obs.track_id);
            if (landmark_it == landmarks_.end() || !landmark_it->second.valid) {
                continue;
            }
            const double error = reprojectionErrorPx(
                intrinsics_,
                extrinsics_,
                state,
                landmark_it->second,
                obs.uv
            );
            if (!std::isfinite(error) || error > params_.max_reprojection_error_px) {
                obs.valid = false;
                ++rejected;
                continue;
            }
            ++support[obs.track_id];
        }
    }

    for (auto& [track_id, landmark] : landmarks_) {
        if (landmark.valid && support[track_id] < 2) {
            landmark.valid = false;
        }
    }

    return rejected;
}

SlidingWindowEstimatorResult SlidingWindowEstimator::addFrame(
    const TrackedFrame& frame,
    const std::optional<PreintegratedImuMeasurement>& imu_factor,
    const std::vector<Landmark>& landmarks,
    const NavState& predicted_state
) {
    seedLandmarks(landmarks);

    WindowState state;
    state.frame_id = frame.state.frame_id;
    state.timestamp = predicted_state.timestamp;
    state.q_wb[0] = predicted_state.q_wb.w();
    state.q_wb[1] = predicted_state.q_wb.x();
    state.q_wb[2] = predicted_state.q_wb.y();
    state.q_wb[3] = predicted_state.q_wb.z();
    state.p_wb[0] = predicted_state.p_wb.x();
    state.p_wb[1] = predicted_state.p_wb.y();
    state.p_wb[2] = predicted_state.p_wb.z();
    state.v_wb[0] = predicted_state.v_wb.x();
    state.v_wb[1] = predicted_state.v_wb.y();
    state.v_wb[2] = predicted_state.v_wb.z();
    state.a_wb[0] = predicted_state.a_wb.x();
    state.a_wb[1] = predicted_state.a_wb.y();
    state.a_wb[2] = predicted_state.a_wb.z();
    state.gyro_bias[0] = predicted_state.gyro_bias.x();
    state.gyro_bias[1] = predicted_state.gyro_bias.y();
    state.gyro_bias[2] = predicted_state.gyro_bias.z();
    state.accel_bias[0] = predicted_state.accel_bias.x();
    state.accel_bias[1] = predicted_state.accel_bias.y();
    state.accel_bias[2] = predicted_state.accel_bias.z();

    states_.push_back(state);
    window_.push_back(frame);
    if (states_.size() > 1) {
        imu_factors_.push_back(imu_factor.value_or(PreintegratedImuMeasurement{}));
    }

    SlidingWindowEstimatorResult result;
    result.state = navStateFromWindowState(states_.back());
    result.camera_pose = cameraFrameFromNavState(frame.state.frame_id, result.state, extrinsics_);
    result.status = status_;

    const int visual_residuals = countVisualResiduals();
    result.visual_residuals = visual_residuals;

    if (!metric_initialized_) {
        dropOldInitializationItems();
        result.state = navStateFromWindowState(states_.back());
        result.camera_pose = cameraFrameFromNavState(
            frame.state.frame_id,
            result.state,
            extrinsics_
        );
        result.visual_residuals = countVisualResiduals();

        if (!hasMetricInitializationWindow()) {
            status_ = states_.empty()
                ? SlidingWindowEstimatorStatus::Uninitialized
                : SlidingWindowEstimatorStatus::Initializing;
            result.status = status_;
            return result;
        }

        const VisualInertialInitializationResult init =
            VisualInertialInitializer::initialize(
                states_,
                window_,
                imu_factors_,
                landmarks_,
                extrinsics_,
                gravity_,
                params_
            );

        if (!init.success) {
            status_ = SlidingWindowEstimatorStatus::Initializing;
            result.status = status_;
            return result;
        }

        std::cout << "[VIO init] metric scale=" << init.scale
                  << " rmse=" << init.rmse
                  << " gravity=" << init.gravity.transpose()
                  << " bg=" << init.gyro_bias.transpose()
                  << " ba=" << init.accel_bias.transpose()
                  << std::endl;

        states_ = init.states;
        window_ = init.frames;
        imu_factors_ = init.imu_factors;
        landmarks_ = init.landmarks;
        metric_initialized_ = true;
        dropOldWindowItems();

        status_ = SlidingWindowEstimatorStatus::Initialized;
        result.success = true;
        result.status = status_;
        result.metric_initialized = true;
        result.estimated_scale = init.scale;
        result.initialization_rmse = init.rmse;
        result.state = navStateFromWindowState(states_.back());
        result.camera_pose = cameraFrameFromNavState(
            states_.back().frame_id,
            result.state,
            extrinsics_
        );
        applyInitializedLandmarksToResult(result);
        return result;
    }

    dropOldWindowItems();

    if (!hasUsableInitialization()) {
        status_ = SlidingWindowEstimatorStatus::Initialized;
        result.status = status_;
        return result;
    }

    ceres::Problem problem;
    for (WindowState& item : states_) {
        problem.AddParameterBlock(item.q_wb, 4, new ceres::QuaternionManifold());
        problem.AddParameterBlock(item.p_wb, 3);
        problem.AddParameterBlock(item.v_wb, 3);
        problem.AddParameterBlock(item.gyro_bias, 3);
        problem.AddParameterBlock(item.accel_bias, 3);
    }

    WindowState& anchor = states_.front();
    ceres::CostFunction* prior_cost =
        new ceres::AutoDiffCostFunction<StatePriorResidual, 15, 4, 3, 3, 3, 3>(
            new StatePriorResidual(anchor, params_)
        );
    problem.AddResidualBlock(
        prior_cost,
        nullptr,
        anchor.q_wb,
        anchor.p_wb,
        anchor.v_wb,
        anchor.gyro_bias,
        anchor.accel_bias
    );

    for (auto& [track_id, landmark] : landmarks_) {
        if (landmark.valid) {
            problem.AddParameterBlock(landmark.p_w, 3);
        }
    }

    for (std::size_t i = 0; i + 1 < states_.size() && i < imu_factors_.size(); ++i) {
        const PreintegratedImuMeasurement& factor = imu_factors_[i];
        if (!factor.valid || factor.dt <= 0.0) {
            continue;
        }
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ImuResidual, 15, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3>(
                new ImuResidual(factor, gravity_, params_)
            );
        problem.AddResidualBlock(
            cost,
            nullptr,
            states_[i].q_wb,
            states_[i].p_wb,
            states_[i].v_wb,
            states_[i].gyro_bias,
            states_[i].accel_bias,
            states_[i + 1].q_wb,
            states_[i + 1].p_wb,
            states_[i + 1].v_wb,
            states_[i + 1].gyro_bias,
            states_[i + 1].accel_bias
        );
    }

    for (std::size_t frame_idx = 0; frame_idx < window_.size(); ++frame_idx) {
        WindowState& frame_state = states_[frame_idx];
        for (const Observation& obs : window_[frame_idx].observations) {
            if (!obs.valid || obs.track_id < 0 || !obs.uv.allFinite()) {
                continue;
            }
            auto landmark_it = landmarks_.find(obs.track_id);
            if (landmark_it == landmarks_.end() || !landmark_it->second.valid) {
                continue;
            }
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 4, 3, 3>(
                    new ReprojectionResidual(
                        intrinsics_,
                        extrinsics_,
                        obs.uv,
                        params_.reprojection_sigma_px
                    )
                );
            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(3.0),
                frame_state.q_wb,
                frame_state.p_wb,
                landmark_it->second.p_w
            );
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = params_.max_solver_iterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (WindowState& item : states_) {
        Eigen::Quaterniond q(item.q_wb[0], item.q_wb[1], item.q_wb[2], item.q_wb[3]);
        q.normalize();
        item.q_wb[0] = q.w();
        item.q_wb[1] = q.x();
        item.q_wb[2] = q.y();
        item.q_wb[3] = q.z();
    }

    ImuPreintegrator repropagator(gravity_);
    for (std::size_t i = 0; i < imu_factors_.size() && i + 1 < states_.size(); ++i) {
        PreintegratedImuMeasurement& factor = imu_factors_[i];
        if (!factor.valid || factor.samples.size() < 2) {
            continue;
        }

        const WindowState& state = states_[i];
        const Eigen::Vector3d gyro_bias(
            state.gyro_bias[0],
            state.gyro_bias[1],
            state.gyro_bias[2]
        );
        const Eigen::Vector3d accel_bias(
            state.accel_bias[0],
            state.accel_bias[1],
            state.accel_bias[2]
        );
        const double bias_delta =
            (gyro_bias - factor.linearization_gyro_bias).norm() +
            (accel_bias - factor.linearization_accel_bias).norm();

        if (bias_delta > params_.bias_repropagation_threshold) {
            factor = repropagator.integrate(
                factor.samples,
                factor.start_time,
                factor.end_time,
                gyro_bias,
                accel_bias
            );
        }
    }

    result.success = summary.IsSolutionUsable();
    result.final_cost = summary.final_cost;
    result.state = navStateFromWindowState(states_.back());
    result.camera_pose = cameraFrameFromNavState(frame.state.frame_id, result.state, extrinsics_);
    result.rejected_observations = rejectReprojectionOutliers();
    status_ = result.success
        ? SlidingWindowEstimatorStatus::Optimized
        : SlidingWindowEstimatorStatus::Degraded;
    result.status = status_;

    std::unordered_map<int, int> observation_counts;
    for (const TrackedFrame& window_frame : window_) {
        for (const Observation& obs : window_frame.observations) {
            if (obs.valid && obs.track_id >= 0) {
                ++observation_counts[obs.track_id];
            }
        }
    }

    result.landmarks.reserve(landmarks_.size());
    int landmark_id = 0;
    for (const auto& [track_id, landmark] : landmarks_) {
        if (!landmark.valid) {
            continue;
        }
        Landmark out;
        out.id = landmark_id++;
        out.track_id = track_id;
        out.p_w = Eigen::Vector3d(landmark.p_w[0], landmark.p_w[1], landmark.p_w[2]);
        out.valid = out.p_w.allFinite();
        out.reprojection_error = 0.0;
        out.num_observations = observation_counts[track_id];
        result.landmarks.push_back(out);
    }

    return result;
}

} // namespace vio
