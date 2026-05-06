#include "estimation/visual_inertial_initializer.hpp"

#include "imu/imu_preintegration.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>

namespace vio {
namespace {

using WindowState = SlidingWindowEstimator::WindowState;
using LandmarkState = SlidingWindowEstimator::LandmarkState;

Eigen::Quaterniond quaternionFromState(const WindowState& state)
{
    Eigen::Quaterniond q(
        state.q_wb[0],
        state.q_wb[1],
        state.q_wb[2],
        state.q_wb[3]
    );
    q.normalize();
    return q;
}

Eigen::Vector3d averageLinearizationBias(
    const std::vector<PreintegratedImuMeasurement>& factors,
    bool gyro
) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int count = 0;
    for (const PreintegratedImuMeasurement& factor : factors) {
        if (!factor.valid) {
            continue;
        }
        sum += gyro ? factor.linearization_gyro_bias
                    : factor.linearization_accel_bias;
        ++count;
    }
    if (count == 0) {
        return Eigen::Vector3d::Zero();
    }
    return sum / static_cast<double>(count);
}

bool solveLeastSquares(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double min_reciprocal_condition,
    Eigen::VectorXd& x,
    double& reciprocal_condition
) {
    if (A.rows() < A.cols() || A.cols() == 0 || b.rows() != A.rows()) {
        return false;
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        A,
        Eigen::ComputeThinU | Eigen::ComputeThinV
    );
    if (svd.singularValues().size() == 0) {
        return false;
    }

    const double sigma_max = svd.singularValues()(0);
    const double sigma_min =
        svd.singularValues()(svd.singularValues().size() - 1);
    if (sigma_max <= 0.0 || sigma_min <= 0.0) {
        return false;
    }

    reciprocal_condition = sigma_min / sigma_max;
    if (reciprocal_condition < min_reciprocal_condition) {
        return false;
    }

    x = svd.solve(b);
    return x.allFinite();
}

bool estimateGyroBias(
    const std::vector<WindowState>& states,
    const std::vector<PreintegratedImuMeasurement>& factors,
    const SlidingWindowEstimatorParams& params,
    Eigen::Vector3d& gyro_bias
) {
    const std::size_t count = std::min(
        factors.size(),
        states.size() > 0 ? states.size() - 1 : std::size_t{0}
    );
    Eigen::MatrixXd A(3 * static_cast<int>(count), 3);
    Eigen::VectorXd b(3 * static_cast<int>(count));

    int row = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const PreintegratedImuMeasurement& factor = factors[i];
        if (!factor.valid) {
            continue;
        }

        const Eigen::Quaterniond q_i = quaternionFromState(states[i]);
        const Eigen::Quaterniond q_j = quaternionFromState(states[i + 1]);
        const Eigen::Quaterniond q_ij_visual = q_i.conjugate() * q_j;
        const Eigen::Quaterniond residual_q =
            factor.delta_q.conjugate().normalized() * q_ij_visual.normalized();
        const Eigen::Vector3d residual = 2.0 * residual_q.vec();

        A.block<3, 3>(row, 0) = factor.jacobian_q_bias_gyro;
        b.segment<3>(row) =
            residual + factor.jacobian_q_bias_gyro * factor.linearization_gyro_bias;
        row += 3;
    }

    if (row < 3) {
        return false;
    }
    A.conservativeResize(row, Eigen::NoChange);
    b.conservativeResize(row);

    Eigen::VectorXd x;
    double reciprocal_condition = 0.0;
    if (!solveLeastSquares(
            A,
            b,
            params.init_min_reciprocal_condition,
            x,
            reciprocal_condition)) {
        return false;
    }

    gyro_bias = x.head<3>();
    return gyro_bias.allFinite();
}

std::vector<PreintegratedImuMeasurement> repropagateFactors(
    const std::vector<PreintegratedImuMeasurement>& factors,
    const Eigen::Vector3d& gyro_bias,
    const Eigen::Vector3d& accel_bias
) {
    std::vector<PreintegratedImuMeasurement> out;
    out.reserve(factors.size());
    ImuPreintegrator repropagator;
    for (const PreintegratedImuMeasurement& factor : factors) {
        if (!factor.valid || factor.samples.size() < 2) {
            out.push_back(factor);
            continue;
        }
        out.push_back(repropagator.integrate(
            factor.samples,
            factor.start_time,
            factor.end_time,
            gyro_bias,
            accel_bias
        ));
    }
    return out;
}

bool solveScaleGravityVelocity(
    const std::vector<WindowState>& states,
    const std::vector<TrackedFrame>& frames,
    const std::vector<PreintegratedImuMeasurement>& factors,
    const CameraImuExtrinsics& extrinsics,
    const SlidingWindowEstimatorParams& params,
    std::vector<Eigen::Vector3d>& velocities,
    Eigen::Vector3d& gravity,
    double& scale,
    double& rmse
) {
    const std::size_t n = states.size();
    const std::size_t factor_count = std::min(factors.size(), n > 0 ? n - 1 : 0);
    const int unknowns = static_cast<int>(3 * n + 4);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(
        6 * static_cast<int>(factor_count),
        unknowns
    );
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * static_cast<int>(factor_count));

    const Eigen::Vector3d t_bc = extrinsics.t_BC();
    int row = 0;
    for (std::size_t i = 0; i < factor_count; ++i) {
        const PreintegratedImuMeasurement& factor = factors[i];
        if (!factor.valid || factor.dt <= 0.0) {
            continue;
        }

        const Eigen::Quaterniond q_i = quaternionFromState(states[i]);
        const Eigen::Quaterniond q_j = quaternionFromState(states[i + 1]);
        const Eigen::Matrix3d R_i = q_i.toRotationMatrix();
        const Eigen::Matrix3d R_j = q_j.toRotationMatrix();
        const Eigen::Matrix3d R_i_t = R_i.transpose();
        const double dt = factor.dt;
        const Eigen::Vector3d camera_delta =
            frames[i + 1].state.t_wc - frames[i].state.t_wc;
        const Eigen::Vector3d body_lever_delta =
            -R_j * t_bc + R_i * t_bc;

        A.block<3, 3>(row, 3 * static_cast<int>(i)) =
            -R_i_t * dt;
        A.block<3, 3>(row, 3 * static_cast<int>(n)) =
            R_i_t * (0.5 * dt * dt);
        A.block<3, 1>(row, 3 * static_cast<int>(n) + 3) =
            R_i_t * camera_delta;
        b.segment<3>(row) =
            factor.delta_p - R_i_t * body_lever_delta;
        row += 3;

        A.block<3, 3>(row, 3 * static_cast<int>(i)) = -R_i_t;
        A.block<3, 3>(row, 3 * static_cast<int>(i + 1)) = R_i_t;
        A.block<3, 3>(row, 3 * static_cast<int>(n)) =
            -R_i_t * dt;
        b.segment<3>(row) = factor.delta_v;
        row += 3;
    }

    if (row < unknowns) {
        return false;
    }
    A.conservativeResize(row, Eigen::NoChange);
    b.conservativeResize(row);

    Eigen::VectorXd x;
    double reciprocal_condition = 0.0;
    if (!solveLeastSquares(
            A,
            b,
            params.init_min_reciprocal_condition,
            x,
            reciprocal_condition)) {
        return false;
    }

    velocities.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        velocities[i] = x.segment<3>(3 * static_cast<int>(i));
    }
    gravity = x.segment<3>(3 * static_cast<int>(n));
    scale = x(3 * static_cast<int>(n) + 3);

    if (!std::isfinite(scale) ||
        scale < params.init_min_scale ||
        scale > params.init_max_scale ||
        !gravity.allFinite()) {
        return false;
    }

    rmse = std::sqrt((A * x - b).squaredNorm() / static_cast<double>(row));
    return std::isfinite(rmse) && rmse <= params.init_max_alignment_rmse;
}

bool estimateAccelBias(
    const std::vector<WindowState>& states,
    const std::vector<Eigen::Vector3d>& p_wb,
    const std::vector<Eigen::Vector3d>& velocities,
    const std::vector<PreintegratedImuMeasurement>& factors,
    const Eigen::Vector3d& gravity,
    const SlidingWindowEstimatorParams& params,
    Eigen::Vector3d& accel_bias
) {
    const std::size_t count = std::min(
        factors.size(),
        states.size() > 0 ? states.size() - 1 : std::size_t{0}
    );
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6 * static_cast<int>(count), 3);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * static_cast<int>(count));

    int row = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const PreintegratedImuMeasurement& factor = factors[i];
        if (!factor.valid || factor.dt <= 0.0) {
            continue;
        }

        const Eigen::Matrix3d R_i_t =
            quaternionFromState(states[i]).toRotationMatrix().transpose();
        const double dt = factor.dt;
        const Eigen::Vector3d predicted_p =
            R_i_t * (p_wb[i + 1] - p_wb[i] - velocities[i] * dt +
                     0.5 * gravity * dt * dt);
        const Eigen::Vector3d predicted_v =
            R_i_t * (velocities[i + 1] - velocities[i] - gravity * dt);

        A.block<3, 3>(row, 0) = factor.jacobian_p_bias_accel;
        b.segment<3>(row) =
            predicted_p - factor.delta_p +
            factor.jacobian_p_bias_accel * factor.linearization_accel_bias;
        row += 3;

        A.block<3, 3>(row, 0) = factor.jacobian_v_bias_accel;
        b.segment<3>(row) =
            predicted_v - factor.delta_v +
            factor.jacobian_v_bias_accel * factor.linearization_accel_bias;
        row += 3;
    }

    if (row < 3) {
        accel_bias = averageLinearizationBias(factors, false);
        return true;
    }
    A.conservativeResize(row, Eigen::NoChange);
    b.conservativeResize(row);

    Eigen::VectorXd x;
    double reciprocal_condition = 0.0;
    if (!solveLeastSquares(
            A,
            b,
            params.init_min_reciprocal_condition,
            x,
            reciprocal_condition)) {
        return false;
    }

    accel_bias = x.head<3>();
    return accel_bias.allFinite() &&
           accel_bias.norm() <= params.init_max_accel_bias_norm;
}

} // namespace

VisualInertialInitializationResult VisualInertialInitializer::initialize(
    const std::vector<WindowState>& states,
    const std::vector<TrackedFrame>& frames,
    const std::vector<PreintegratedImuMeasurement>& imu_factors,
    const std::unordered_map<int, LandmarkState>& landmarks,
    const CameraImuExtrinsics& extrinsics,
    const Eigen::Vector3d& gravity_reference,
    const SlidingWindowEstimatorParams& params
) {
    VisualInertialInitializationResult result;
    result.states = states;
    result.frames = frames;
    result.imu_factors = imu_factors;
    result.landmarks = landmarks;

    if (states.size() < static_cast<std::size_t>(params.init_min_frames) ||
        frames.size() != states.size() ||
        imu_factors.size() + 1 < states.size()) {
        result.failure_reason = "not enough aligned visual/IMU states";
        return result;
    }

    Eigen::Vector3d gyro_bias = averageLinearizationBias(imu_factors, true);
    if (!estimateGyroBias(states, imu_factors, params, gyro_bias)) {
        result.failure_reason = "gyro bias solve failed";
        return result;
    }

    Eigen::Vector3d accel_bias = averageLinearizationBias(imu_factors, false);
    std::vector<PreintegratedImuMeasurement> corrected_factors =
        repropagateFactors(imu_factors, gyro_bias, accel_bias);

    std::vector<Eigen::Vector3d> velocities;
    Eigen::Vector3d solved_gravity = gravity_reference;
    double scale = 1.0;
    double rmse = std::numeric_limits<double>::infinity();
    if (!solveScaleGravityVelocity(
            states,
            frames,
            corrected_factors,
            extrinsics,
            params,
            velocities,
            solved_gravity,
            scale,
            rmse)) {
        result.failure_reason = "scale/gravity/velocity solve failed";
        return result;
    }

    const double gravity_norm = gravity_reference.norm();
    if (gravity_norm <= 0.0 ||
        std::abs(solved_gravity.norm() - gravity_norm) >
            params.init_gravity_norm_tolerance) {
        result.failure_reason = "gravity norm is inconsistent";
        return result;
    }

    Eigen::Quaterniond q_align = Eigen::Quaterniond::FromTwoVectors(
        solved_gravity.normalized(),
        gravity_reference.normalized()
    );
    q_align.normalize();
    const Eigen::Matrix3d R_align = q_align.toRotationMatrix();

    std::vector<Eigen::Vector3d> p_wb(states.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> aligned_velocities(
        states.size(),
        Eigen::Vector3d::Zero()
    );
    for (std::size_t i = 0; i < states.size(); ++i) {
        const Eigen::Quaterniond q_wc_old = frames[i].state.q_wc.normalized();
        const Eigen::Quaterniond q_wc_new =
            (q_align * q_wc_old).normalized();
        const Eigen::Matrix3d R_wc_new = q_wc_new.toRotationMatrix();
        const Eigen::Matrix3d R_wb_new =
            R_wc_new * extrinsics.R_BC().transpose();
        const Eigen::Vector3d t_wc_new =
            R_align * (scale * frames[i].state.t_wc);

        result.frames[i].state.q_wc = q_wc_new;
        result.frames[i].state.t_wc = t_wc_new;
        aligned_velocities[i] = R_align * velocities[i];
        result.frames[i].state.v_w = aligned_velocities[i];
        result.frames[i].state.a_w.setZero();

        p_wb[i] = t_wc_new - R_wb_new * extrinsics.t_BC();

        WindowState& state = result.states[i];
        const Eigen::Quaterniond q_wb_new =
            Eigen::Quaterniond(R_wb_new).normalized();
        state.q_wb[0] = q_wb_new.w();
        state.q_wb[1] = q_wb_new.x();
        state.q_wb[2] = q_wb_new.y();
        state.q_wb[3] = q_wb_new.z();
        state.p_wb[0] = p_wb[i].x();
        state.p_wb[1] = p_wb[i].y();
        state.p_wb[2] = p_wb[i].z();
        const Eigen::Vector3d v_wb = aligned_velocities[i];
        state.v_wb[0] = v_wb.x();
        state.v_wb[1] = v_wb.y();
        state.v_wb[2] = v_wb.z();
        state.a_wb[0] = 0.0;
        state.a_wb[1] = 0.0;
        state.a_wb[2] = 0.0;
    }

    const Eigen::Vector3d aligned_gravity = R_align * solved_gravity;
    if (!estimateAccelBias(
            result.states,
            p_wb,
            aligned_velocities,
            corrected_factors,
            aligned_gravity,
            params,
            accel_bias)) {
        result.failure_reason = "accelerometer bias solve failed";
        return result;
    }

    result.imu_factors =
        repropagateFactors(corrected_factors, gyro_bias, accel_bias);

    for (WindowState& state : result.states) {
        state.gyro_bias[0] = gyro_bias.x();
        state.gyro_bias[1] = gyro_bias.y();
        state.gyro_bias[2] = gyro_bias.z();
        state.accel_bias[0] = accel_bias.x();
        state.accel_bias[1] = accel_bias.y();
        state.accel_bias[2] = accel_bias.z();
    }

    for (auto& [track_id, landmark] : result.landmarks) {
        if (!landmark.valid) {
            continue;
        }
        Eigen::Vector3d p_w(landmark.p_w[0], landmark.p_w[1], landmark.p_w[2]);
        if (!p_w.allFinite()) {
            landmark.valid = false;
            continue;
        }
        p_w = R_align * (scale * p_w);
        landmark.p_w[0] = p_w.x();
        landmark.p_w[1] = p_w.y();
        landmark.p_w[2] = p_w.z();
    }

    result.success = true;
    result.scale = scale;
    result.rmse = rmse;
    result.gravity = aligned_gravity;
    result.gyro_bias = gyro_bias;
    result.accel_bias = accel_bias;
    return result;
}

} // namespace vio
