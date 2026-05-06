#include "imu/imu_preintegration.hpp"

#include <algorithm>
#include <cmath>

namespace vio {
namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return m;
}

ImuSample interpolateSample(
    const ImuSample& a,
    const ImuSample& b,
    double timestamp
) {
    if (b.t <= a.t) {
        ImuSample out = a;
        out.t = timestamp;
        return out;
    }

    const double alpha = std::clamp((timestamp - a.t) / (b.t - a.t), 0.0, 1.0);
    ImuSample out;
    out.t = timestamp;
    out.gyro = (1.0 - alpha) * a.gyro + alpha * b.gyro;
    out.acc = (1.0 - alpha) * a.acc + alpha * b.acc;
    return out;
}

std::vector<ImuSample> makeBoundedSamples(
    const std::vector<ImuSample>& imu,
    double start_time,
    double end_time
) {
    std::vector<ImuSample> samples;
    if (imu.empty() || end_time <= start_time) {
        return samples;
    }

    const auto by_time = [](const ImuSample& sample, double timestamp) {
        return sample.t < timestamp;
    };

    auto first_after_start =
        std::lower_bound(imu.begin(), imu.end(), start_time, by_time);
    auto first_after_end =
        std::lower_bound(imu.begin(), imu.end(), end_time, by_time);

    if (first_after_start == imu.end()) {
        return samples;
    }

    if (first_after_start == imu.begin()) {
        ImuSample first = *first_after_start;
        first.t = start_time;
        samples.push_back(first);
    } else {
        samples.push_back(interpolateSample(
            *(first_after_start - 1),
            *first_after_start,
            start_time
        ));
    }

    for (auto it = first_after_start; it != imu.end() && it->t < end_time; ++it) {
        if (it->t > start_time) {
            samples.push_back(*it);
        }
    }

    if (first_after_end == imu.end()) {
        if (imu.back().t >= end_time) {
            samples.push_back(imu.back());
        }
    } else if (first_after_end == imu.begin()) {
        ImuSample last = *first_after_end;
        last.t = end_time;
        samples.push_back(last);
    } else {
        samples.push_back(interpolateSample(
            *(first_after_end - 1),
            *first_after_end,
            end_time
        ));
    }

    return samples;
}

} // namespace

ImuPreintegrator::ImuPreintegrator(const Eigen::Vector3d& gravity)
    : gravity_(gravity)
{
}

PreintegratedImuMeasurement ImuPreintegrator::integrate(
    const std::vector<ImuSample>& imu,
    double start_time,
    double end_time,
    const Eigen::Vector3d& gyro_bias,
    const Eigen::Vector3d& accel_bias
) const {
    PreintegratedImuMeasurement result;
    result.start_time = start_time;
    result.end_time = end_time;
    result.dt = end_time - start_time;
    result.linearization_gyro_bias = gyro_bias;
    result.linearization_accel_bias = accel_bias;
    result.covariance = Eigen::Matrix<double, 15, 15>::Identity() * 1e-6;

    std::vector<ImuSample> samples =
        makeBoundedSamples(imu, start_time, end_time);
    result.samples = samples;

    if (samples.size() < 2 || result.dt <= 0.0) {
        return result;
    }

    Eigen::Quaterniond q = Eigen::Quaterniond(1, 0, 0, 0);
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    double accumulated_dt = 0.0;
    Eigen::Matrix3d jp_ba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jv_ba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jp_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jv_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d jq_bg = Eigen::Matrix3d::Zero();

    for (std::size_t i = 1; i < samples.size(); ++i) {
        const ImuSample& prev = samples[i - 1];
        const ImuSample& curr = samples[i];
        const double dt = curr.t - prev.t;
        if (dt <= 0.0) {
            continue;
        }

        const Eigen::Vector3d gyro_corr =
            0.5 * (prev.gyro + curr.gyro) - gyro_bias;
        const Eigen::Vector3d acc_corr =
            0.5 * (prev.acc + curr.acc) - accel_bias;

        const Eigen::Quaterniond q_prev = q;
        const Eigen::Matrix3d R_prev = q_prev.toRotationMatrix();
        const Eigen::Vector3d acc_i = q_prev * acc_corr;

        const Eigen::Matrix3d acc_skew = skew(acc_corr);
        jp_ba += jv_ba * dt - 0.5 * R_prev * dt * dt;
        jv_ba += -R_prev * dt;
        jp_bg += jv_bg * dt - 0.5 * R_prev * acc_skew * jq_bg * dt * dt;
        jv_bg += -R_prev * acc_skew * jq_bg * dt;
        jq_bg += -Eigen::Matrix3d::Identity() * dt;

        q *= deltaQuat(gyro_corr, dt);
        q.normalize();

        const Eigen::Vector3d v_prev = v;
        v += acc_i * dt;
        p += v_prev * dt + 0.5 * acc_i * dt * dt;
        accumulated_dt += dt;

        ++result.num_samples;
    }

    result.delta_q = q.normalized();
    result.delta_v = v;
    result.delta_p = p;
    result.jacobian_v_bias_accel = jv_ba;
    result.jacobian_p_bias_accel = jp_ba;
    result.jacobian_v_bias_gyro = jv_bg;
    result.jacobian_p_bias_gyro = jp_bg;
    result.jacobian_q_bias_gyro = jq_bg;
    const double gyro_noise = 1.7e-4;
    const double accel_noise = 2.0e-3;
    const double bias_noise = 1.0e-5;
    result.covariance.setZero();
    result.covariance.block<3, 3>(0, 0).setIdentity();
    result.covariance.block<3, 3>(3, 3).setIdentity();
    result.covariance.block<3, 3>(6, 6).setIdentity();
    result.covariance.block<3, 3>(9, 9).setIdentity();
    result.covariance.block<3, 3>(12, 12).setIdentity();
    const double safe_dt = std::max(result.dt, 1e-3);
    result.covariance.block<3, 3>(0, 0) *= accel_noise * accel_noise * safe_dt * safe_dt * safe_dt / 3.0;
    result.covariance.block<3, 3>(3, 3) *= accel_noise * accel_noise * safe_dt;
    result.covariance.block<3, 3>(6, 6) *= gyro_noise * gyro_noise * safe_dt;
    result.covariance.block<3, 3>(9, 9) *= bias_noise * bias_noise * safe_dt;
    result.covariance.block<3, 3>(12, 12) *= bias_noise * bias_noise * safe_dt;
    result.valid = result.num_samples > 0;
    return result;
}

} // namespace vio
