#pragma once

#include "imu/imu_processor.hpp"

#include <vector>

namespace vio {

struct PreintegratedImuMeasurement {
    double start_time = 0.0;
    double end_time = 0.0;
    double dt = 0.0;
    std::vector<ImuSample> samples;
    Eigen::Quaterniond delta_q = Eigen::Quaterniond(1, 0, 0, 0);
    Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
    Eigen::Vector3d linearization_gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d linearization_accel_bias = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 15, 15> covariance =
        Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix3d jacobian_p_bias_accel = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jacobian_p_bias_gyro = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jacobian_v_bias_accel = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jacobian_v_bias_gyro = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jacobian_q_bias_gyro = Eigen::Matrix3d::Zero();
    std::size_t num_samples = 0;
    bool valid = false;
};

class ImuPreintegrator {
public:
    explicit ImuPreintegrator(
        const Eigen::Vector3d& gravity = Eigen::Vector3d(0.0, 0.0, 9.81)
    );

    [[nodiscard]] PreintegratedImuMeasurement integrate(
        const std::vector<ImuSample>& imu,
        double start_time,
        double end_time,
        const Eigen::Vector3d& gyro_bias,
        const Eigen::Vector3d& accel_bias
    ) const;

private:
    Eigen::Vector3d gravity_;
};

} // namespace vio
