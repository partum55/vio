#include "imu/imu_processor.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace vio {

namespace
{
    constexpr double init_duration = 3.0;
    constexpr double epsilon = 1e-12;

    struct ImuInitResult
    {
        Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d avg_gyro = Eigen::Vector3d::Zero();
        std::size_t start_idx = 0;
        std::size_t end_idx = 0;
        bool valid = false;
    };

    std::size_t findStartIndex(const std::vector<ImuSample>& imu, double t0)
    {
        std::size_t idx = 0;

        while (idx < imu.size() && imu[idx].t < t0) {
            ++idx;
        }

        return idx;
    }

    ImuInitResult estimateInitialImuState(
        const std::vector<ImuSample>& imu,
        double t0,
        double init_duration_sec
    )
    {
        ImuInitResult result;

        if (imu.empty()) {
            return result;
        }

        const std::size_t start_idx = findStartIndex(imu, t0);

        if (start_idx >= imu.size()) {
            return result;
        }

        const double t_end = imu[start_idx].t + init_duration_sec;

        Eigen::Vector3d acc_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();

        std::size_t count = 0;
        std::size_t end_idx = start_idx;

        while (end_idx < imu.size() && imu[end_idx].t <= t_end) {
            acc_sum += imu[end_idx].acc;
            gyro_sum += imu[end_idx].gyro;

            ++count;
            ++end_idx;
        }

        if (count == 0) {
            return result;
        }

        result.avg_acc = acc_sum / static_cast<double>(count);
        result.avg_gyro = gyro_sum / static_cast<double>(count);
        result.start_idx = start_idx;
        result.end_idx = end_idx;
        result.valid = true;

        return result;
    }
}

Eigen::Quaterniond deltaQuat(const Eigen::Vector3d& omega, double dt)
{
    const double angle = omega.norm() * dt;

    if (angle < epsilon) {
        return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    }

    const Eigen::Vector3d axis = omega.normalized();

    Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, axis));
    dq.normalize();

    return dq;
}

Eigen::Quaterniond initialOrientationFromAccel(
    const Eigen::Vector3d& acc_meas,
    const Eigen::Vector3d& gravity_world
)
{
    if (acc_meas.norm() < epsilon || gravity_world.norm() < epsilon) {
        return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    }

    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(
        acc_meas.normalized(),
        gravity_world.normalized()
    );

    q.normalize();
    return q;
}

class Kalman
{
public:
    Eigen::Vector2d mu;
    Eigen::Matrix2d Sigma;

    double sigma_meas;
    double q_process;

    Kalman(double sigma_meas_ = 0.01, double q_process_ = 1.0)
        : sigma_meas(sigma_meas_),
          q_process(q_process_)
    {
        mu << 0.0, 0.0;
        Sigma << 1.0, 0.0, 0.0, 1.0;
    }

    void init(double first_measurement)
    {
        mu << first_measurement, 0.0;

        Sigma << sigma_meas * sigma_meas, 0.0, 0.0, 1.0;
    }

    void predict(double dt)
    {
        Eigen::Matrix2d A;
        A << 1.0, dt, 0.0, 1.0;

        Eigen::Matrix2d Q;
        Q << dt * dt * dt / 3.0, dt * dt / 2.0, dt * dt / 2.0, dt;

        Q *= q_process;

        mu = A * mu;
        Sigma = A * Sigma * A.transpose() + Q;
    }

    double update(double z_meas)
    {
        Eigen::RowVector2d H;
        H << 1.0, 0.0;

        const double R = sigma_meas * sigma_meas;
        const double innovation = z_meas - (H * mu)(0);
        const double S = (H * Sigma * H.transpose())(0, 0) + R;

        const Eigen::Vector2d K = Sigma * H.transpose() / S;

        mu = mu + K * innovation;

        const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        Sigma = (I - K * H) * Sigma * (I - K * H).transpose() + K * R * K.transpose();

        return mu(0);
    }
};

ImuProcessor::ImuProcessor(const Eigen::Vector3d& gravity)
    : gravity_(gravity)
{
}

bool ImuProcessor::initialize(
    const std::vector<ImuSample>& imu_data,
    double start_time,
    double init_duration_sec
)
{
    imu_ = imu_data;
    initialized_ = false;
    current_idx_ = 0;
    pose_ = Pose{};

    if (imu_.empty()) {
        std::cerr << "IMU initialization failed: empty IMU data\n";
        return false;
    }

    std::sort(
        imu_.begin(),
        imu_.end(),
        [](const ImuSample& a, const ImuSample& b) {
            return a.t < b.t;
        }
    );

    const ImuInitResult init = estimateInitialImuState(
        imu_,
        start_time,
        init_duration_sec
    );

    if (!init.valid) {
        std::cerr << "IMU initialization failed: invalid initialization window\n";
        return false;
    }

    current_idx_ = init.end_idx;

    if (current_idx_ >= imu_.size()) {
        std::cerr << "IMU initialization failed: not enough IMU samples after initialization\n";
        return false;
    }

    pose_.t = imu_[current_idx_].t;
    pose_.p.setZero();
    pose_.v.setZero();

    pose_.q = initialOrientationFromAccel(init.avg_acc, gravity_);
    pose_.q.normalize();

    gyro_bias_ = init.avg_gyro;

    const Eigen::Vector3d gravity_body_expected = pose_.q.conjugate() * gravity_;

    accel_bias_ = init.avg_acc - gravity_body_expected;

    initialized_ = true;

    return true;
}

void ImuProcessor::propagateUntil(double timestamp)
{
    if (!initialized_) {
        std::cerr << "IMU propagateUntil() called before initialize()\n";
        return;
    }

    while (current_idx_ < imu_.size() && imu_[current_idx_].t <= timestamp) {
        const ImuSample& s = imu_[current_idx_];

        const double dt = s.t - pose_.t;

        if (dt <= 0.0) {
            ++current_idx_;
            continue;
        }

        const Eigen::Vector3d gyro_corr = s.gyro - gyro_bias_;
        const Eigen::Vector3d acc_corr = s.acc - accel_bias_;

        pose_.q *= deltaQuat(gyro_corr, dt);
        pose_.q.normalize();

        const Eigen::Vector3d acc_world = pose_.q * acc_corr;
        const Eigen::Vector3d acc_lin = acc_world - gravity_;

        const Eigen::Vector3d v_prev = pose_.v;

        pose_.v += acc_lin * dt;
        pose_.p += v_prev * dt + 0.5 * acc_lin * dt * dt;

        pose_.t = s.t;

        ++current_idx_;
    }
}

Pose ImuProcessor::getCurrentPose() const
{
    return pose_;
}

double ImuProcessor::computeBaseline(const Pose& pivot_pose) const
{
    return (pose_.p - pivot_pose.p).norm();
}

void ImuProcessor::correctVelocityFromVisualDisplacement(
    const Eigen::Vector3d& pivot_visual_position,
    const Eigen::Vector3d& current_visual_position,
    double pivot_timestamp,
    double current_timestamp
)
{
    if (!initialized_) {
        return;
    }

    const double dt = current_timestamp - pivot_timestamp;

    if (dt <= 1e-6 || !std::isfinite(dt)) {
        return;
    }

    const Eigen::Vector3d visual_delta =
        current_visual_position - pivot_visual_position;

    if (!visual_delta.allFinite()) {
        return;
    }

    const Eigen::Vector3d visual_velocity = visual_delta / dt;

    const double visual_speed = visual_velocity.norm();
    const double imu_speed = pose_.v.norm();

    if (!std::isfinite(visual_speed) || !std::isfinite(imu_speed)) {
        return;
    }

    if (visual_speed < 1e-6) {
        return;
    }

    if (imu_speed < 1e-6) {
        pose_.v = visual_velocity;

        std::cout << "IMU velocity correction: replaced zero velocity with visual velocity = "
                  << pose_.v.transpose()
                  << "\n";

        return;
    }

    double velocity_scale = visual_speed / imu_speed;

    velocity_scale = std::clamp(
        velocity_scale,
        0.2,
        2.0
    );

    constexpr double alpha = 0.3;

    const double blended_scale =
        (1.0 - alpha) + alpha * velocity_scale;

    pose_.v *= blended_scale;

    std::cout << "IMU velocity correction: "
              << "visual_speed=" << visual_speed
              << ", imu_speed=" << imu_speed
              << ", scale=" << velocity_scale
              << ", blended_scale=" << blended_scale
              << ", corrected_v=" << pose_.v.transpose()
              << "\n";
}

void integrateImuFiltered(
    const std::vector<ImuSample>& imu,
    double t0,
    double t1,
    Pose& pose,
    const Eigen::Vector3d& gravity,
    std::vector<Pose>& trajectory_out
)
{
    trajectory_out.clear();
    trajectory_out.reserve(imu.size());

    if (imu.empty()) {
        return;
    }

    const ImuInitResult init = estimateInitialImuState(imu, t0, init_duration);

    if (!init.valid) {
        return;
    }

    std::size_t idx = init.end_idx;

    if (idx >= imu.size()) {
        return;
    }

    pose.t = imu[idx].t;
    pose.p.setZero();
    pose.v.setZero();

    pose.q = initialOrientationFromAccel(init.avg_acc, gravity);
    pose.q.normalize();

    const Eigen::Vector3d gyro_bias = init.avg_gyro;

    const Eigen::Vector3d gravity_body_expected =
        pose.q.conjugate() * gravity;

    const Eigen::Vector3d accel_bias =
        init.avg_acc - gravity_body_expected;

    const double gyro_noise_density = 1.6968e-04;
    const double gyro_random_walk = 1.9393e-05;
    const double accel_noise_density = 2.0e-03;
    const double accel_random_walk = 3.0e-03;

    const double rate_hz = 200.0;
    const double gyro_q_scale = 10.0;
    const double accel_q_scale = 10.0;

    const double gyro_sigma = gyro_noise_density * std::sqrt(rate_hz);
    const double accel_sigma = accel_noise_density * std::sqrt(rate_hz);

    const double gyro_q = gyro_q_scale * gyro_random_walk * gyro_random_walk;
    const double accel_q = accel_q_scale * accel_random_walk * accel_random_walk;

    Kalman kf_gx(gyro_sigma, gyro_q);
    Kalman kf_gy(gyro_sigma, gyro_q);
    Kalman kf_gz(gyro_sigma, gyro_q);

    Kalman kf_ax(accel_sigma, accel_q);
    Kalman kf_ay(accel_sigma, accel_q);
    Kalman kf_az(accel_sigma, accel_q);

    const Eigen::Vector3d gyro0 = imu[idx].gyro - gyro_bias;
    const Eigen::Vector3d acc0 = imu[idx].acc - accel_bias;

    kf_gx.init(gyro0.x());
    kf_gy.init(gyro0.y());
    kf_gz.init(gyro0.z());

    kf_ax.init(acc0.x());
    kf_ay.init(acc0.y());
    kf_az.init(acc0.z());

    while (idx < imu.size() && imu[idx].t <= t1) {
        const ImuSample& s = imu[idx];

        const double dt = s.t - pose.t;

        if (dt <= 0.0) {
            ++idx;
            continue;
        }

        kf_gx.predict(dt);
        kf_gy.predict(dt);
        kf_gz.predict(dt);

        kf_ax.predict(dt);
        kf_ay.predict(dt);
        kf_az.predict(dt);

        const Eigen::Vector3d gyro_meas = s.gyro - gyro_bias;
        const Eigen::Vector3d acc_meas = s.acc - accel_bias;

        Eigen::Vector3d gyro_filtered;
        Eigen::Vector3d acc_filtered;

        gyro_filtered.x() = kf_gx.update(gyro_meas.x());
        gyro_filtered.y() = kf_gy.update(gyro_meas.y());
        gyro_filtered.z() = kf_gz.update(gyro_meas.z());

        acc_filtered.x() = kf_ax.update(acc_meas.x());
        acc_filtered.y() = kf_ay.update(acc_meas.y());
        acc_filtered.z() = kf_az.update(acc_meas.z());

        pose.q *= deltaQuat(gyro_filtered, dt);
        pose.q.normalize();

        const Eigen::Vector3d acc_world = pose.q * acc_filtered;
        const Eigen::Vector3d acc_lin = acc_world - gravity;

        const Eigen::Vector3d v_prev = pose.v;

        pose.v += acc_lin * dt;
        pose.p += v_prev * dt + 0.5 * acc_lin * dt * dt;

        pose.t = s.t;

        trajectory_out.push_back(pose);

        ++idx;
    }
}

void integrateImuRaw(
    const std::vector<ImuSample>& imu,
    double t0,
    double t1,
    Pose& pose,
    const Eigen::Vector3d& gravity,
    std::vector<Pose>& trajectory_out
)
{
    trajectory_out.clear();
    trajectory_out.reserve(imu.size());

    if (imu.empty()) {
        return;
    }

    const ImuInitResult init = estimateInitialImuState(imu, t0, init_duration);

    if (!init.valid) {
        return;
    }

    std::size_t idx = init.end_idx;

    if (idx >= imu.size()) {
        return;
    }

    pose.t = imu[idx].t;
    pose.p.setZero();
    pose.v.setZero();

    pose.q = initialOrientationFromAccel(init.avg_acc, gravity);
    pose.q.normalize();

    const Eigen::Vector3d gyro_bias = init.avg_gyro;

    const Eigen::Vector3d gravity_body_expected =
        pose.q.conjugate() * gravity;

    const Eigen::Vector3d accel_bias =
        init.avg_acc - gravity_body_expected;

    while (idx < imu.size() && imu[idx].t <= t1) {
        const ImuSample& s = imu[idx];

        const double dt = s.t - pose.t;

        if (dt <= 0.0) {
            ++idx;
            continue;
        }

        const Eigen::Vector3d gyro_corr = s.gyro - gyro_bias;
        const Eigen::Vector3d acc_corr = s.acc - accel_bias;

        pose.q *= deltaQuat(gyro_corr, dt);
        pose.q.normalize();

        const Eigen::Vector3d acc_world = pose.q * acc_corr;
        const Eigen::Vector3d acc_lin = acc_world - gravity;

        const Eigen::Vector3d v_prev = pose.v;

        pose.v += acc_lin * dt;
        pose.p += v_prev * dt + 0.5 * acc_lin * dt * dt;

        pose.t = s.t;

        trajectory_out.push_back(pose);

        ++idx;
    }
}

bool loadImuCsv(const std::string& path, std::vector<ImuSample>& out)
{
    out.clear();

    std::ifstream f(path);

    if (!f) {
        std::cerr << "Couldn't open IMU csv: " << path << "\n";
        return false;
    }

    std::string line;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);

        double t_ns = 0.0;
        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;

        char comma = '\0';

        ss >> t_ns >> comma
           >> gx >> comma
           >> gy >> comma
           >> gz >> comma
           >> ax >> comma
           >> ay >> comma
           >> az;

        if (ss.fail()) {
            continue;
        }

        ImuSample s;

        s.t = t_ns * 1e-9;
        s.gyro = Eigen::Vector3d(gx, gy, gz);
        s.acc = Eigen::Vector3d(ax, ay, az);

        out.push_back(s);
    }

    std::sort(
        out.begin(),
        out.end(),
        [](const ImuSample& a, const ImuSample& b) {
            return a.t < b.t;
        }
    );

    return !out.empty();
}

bool saveTrajectoryCsv(const std::string& path, const std::vector<Pose>& traj)
{
    std::ofstream out(path);

    if (!out.is_open()) {
        return false;
    }

    out << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz\n";
    out << std::fixed << std::setprecision(9);

    for (const Pose& p : traj) {
        out << p.t << ","
            << p.p.x() << "," << p.p.y() << "," << p.p.z() << ","
            << p.v.x() << "," << p.v.y() << "," << p.v.z() << ","
            << p.q.w() << "," << p.q.x() << "," << p.q.y() << "," << p.q.z()
            << "\n";
    }

    return true;
}

} // namespace vio