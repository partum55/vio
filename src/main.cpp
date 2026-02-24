#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "imu.hpp"

static void integrateStepByStep(const std::vector<ImuSample>& imu, double t_start, double t_end, Pose& pose, const Eigen::Vector3d& gravity, std::vector<Pose>& trajectory_out)
{
    trajectory_out.clear();
    trajectory_out.reserve(10000);

    size_t i0 = 0;
    while (i0 < imu.size() && imu[i0].t < t_start) i0++;
    if (i0 == imu.size()) return;

    pose.t = imu[i0].t;

    for (size_t i = i0 + 1; i < imu.size(); ++i) {
        if (imu[i].t > t_end) break;

        const double t_prev = imu[i-1].t;
        const double t_cur  = imu[i].t;
        const double dt = t_cur - t_prev;
        if (dt <= 0) continue;

        Eigen::Quaterniond dq = deltaQuat(imu[i].gyro, dt);
        pose.q = pose.q * dq;
        pose.q.normalize();

        const Eigen::Vector3d acc_world = pose.q * imu[i].acc;
        const Eigen::Vector3d acc_lin = acc_world - gravity;

        pose.v += acc_lin * dt;
        pose.p += pose.v * dt;

        pose.t = t_cur;
        trajectory_out.push_back(pose);
    }
}

int main(int argc, char** argv){
    if (argc < 3) {
    std::cerr << "Usage: ./imu_test <imu_csv> <seconds>\n";
    return 1;
    }

    const std::string imu_path = argv[1];
    const double seconds = std::stod(argv[2]);

    std::vector<ImuSample> imu;
    if (!loadImuCsv(imu_path, imu)) {
        std::cerr << "Failed to load csv\n";
        return 2;
    }

    const double t0 = imu.front().t;
    const double t1 = std::min(imu.back().t, t0 + seconds);

    const Eigen::Vector3d gravity(0, 0, 9.81);

    Pose pose;

    std::vector<Pose> trajectory;
    integrateStepByStep(imu, t0, t1, pose, gravity, trajectory);

    if (trajectory.empty()) {
        std::cerr << "Trajectory is empty\n";
        return 3;
    }

    saveTrajectoryCsv("trajectory.csv", trajectory);

    const Pose& last = trajectory.back();

    std::cout << "IMU samples: " << imu.size() << "\n";
    std::cout << "Duration: " << last.t - t0 << " s\n";
    std::cout << "Position: " << last.p.transpose() << "\n";
    std::cout << "Velocity: " << last.v.transpose() << "\n";
    std::cout << "Quaternion q = [" << last.q.w() << ", " << last.q.x() << ", " << last.q.y() << ", " << last.q.z() << "]\n";
    return 0;
}