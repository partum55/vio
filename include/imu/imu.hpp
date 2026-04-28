#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

struct ImuSample {
    double t = 0.0;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero(); // rad/s
    Eigen::Vector3d acc  = Eigen::Vector3d::Zero(); // with gravity
};

struct Pose {
    double t = 0.0;
    Eigen::Quaterniond q = Eigen::Quaterniond(1, 0, 0, 0);
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

class ImuProcessor {
public:
    explicit ImuProcessor(
        const Eigen::Vector3d& gravity = Eigen::Vector3d(0.0, 0.0, -9.81)
    );

    bool initialize(
        const std::vector<ImuSample>& imu_data,
        double start_time,
        double init_duration_sec = 3.0
    );

    void propagateUntil(double timestamp);

    Pose getCurrentPose() const;

    double computeBaseline(const Pose& pivot_pose) const;

private:
    std::vector<ImuSample> imu_;
    Pose pose_;

    Eigen::Vector3d gravity_;
    Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias_ = Eigen::Vector3d::Zero();

    std::size_t current_idx_ = 0;
    bool initialized_ = false;
};

Eigen::Quaterniond deltaQuat(const Eigen::Vector3d& omega, double dt);

Eigen::Quaterniond initialOrientationFromAccel(
    const Eigen::Vector3d& acc_meas,
    const Eigen::Vector3d& gravity_world
);

void integrateImuFiltered(
    const std::vector<ImuSample>& imu,
    double t0,
    double t1,
    Pose& pose,
    const Eigen::Vector3d& gravity,
    std::vector<Pose>& trajectory_out
);

void integrateImuRaw(
    const std::vector<ImuSample>& imu,
    double t0,
    double t1,
    Pose& pose,
    const Eigen::Vector3d& gravity,
    std::vector<Pose>& trajectory_out
);

bool loadImuCsv(const std::string& path, std::vector<ImuSample>& out);

bool saveTrajectoryCsv(const std::string& path, const std::vector<Pose>& traj);