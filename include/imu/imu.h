#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <string>
#include <vector>

namespace vio {

struct ImuSample {
    double t = 0.0;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero(); // rad/s
    Eigen::Vector3d acc = Eigen::Vector3d::Zero(); // with gravity
};

struct ImuPose {
    double t = 0.0;
    Eigen::Quaterniond q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

Eigen::Quaterniond deltaQuat(const Eigen::Vector3d& omega, double dt);

void integrateImuFiltered(const std::vector<ImuSample>& imu,
                          double t0,
                          double t1,
                          ImuPose& pose,
                          const Eigen::Vector3d& gravity,
                          std::vector<ImuPose>& trajectory_out);

void integrateImuRaw(const std::vector<ImuSample>& imu,
                     double t0,
                     double t1,
                     ImuPose& pose,
                     const Eigen::Vector3d& gravity,
                     std::vector<ImuPose>& trajectory_out);

bool loadImuCsv(const std::string& path, std::vector<ImuSample>& out);
bool saveTrajectoryCsv(const std::string& path, const std::vector<ImuPose>& traj);

} // namespace vio
