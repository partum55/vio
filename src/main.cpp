#include <iostream>
#include "imu.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./imu_test <imu_csv> <seconds>\n";
        return 1;
    }

    const std::string imu_path = argv[1];
    const double seconds = std::stod(argv[2]);

    std::vector<ImuSample> imu;
    if (!loadImuCsv(imu_path, imu)) {
        std::cerr << "Failed to load CSV\n";
        return 2;
    }

    const double t0 = imu.front().t;
    const double t1 = std::min(imu.back().t, t0 + seconds);

    Eigen::Vector3d gravity(0,0,9.81);
    Pose pose;
    std::vector<Pose> trajectory;

    integrateImuFiltered(imu, t0, t1, pose, gravity, trajectory);

    if (!trajectory.empty()) {
        saveTrajectoryCsv("trajectory.csv", trajectory);
        const Pose& last = trajectory.back();
        std::cout << "Duration: " << last.t - t0 << " s\n";
        std::cout << "Position: " << last.p.transpose() << "\n";
        std::cout << "Velocity: " << last.v.transpose() << "\n";
        std::cout << "Quaternion: [" << last.q.w() << ", " << last.q.x() << ", " << last.q.y() << ", " << last.q.z() << "]\n";
    }
    return 0;
}