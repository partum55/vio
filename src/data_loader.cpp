#include "vio/data_loader.h"

#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <sstream>

namespace vio {

Trajectory loadTrajectoryTUM(const std::string& filepath) {
    Trajectory trajectory;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: could not open trajectory file: " << filepath << "\n";
        return trajectory;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double ts, tx, ty, tz, qx, qy, qz, qw;
        if (!(iss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            std::cerr << "Warning: skipping malformed line " << line_num
                      << " in " << filepath << "\n";
            continue;
        }

        Eigen::Quaterniond q(qw, qx, qy, qz); // Eigen takes w-first
        q.normalize();

        CameraPose pose;
        pose.timestamp = ts;
        pose.T_wc = Eigen::Matrix4d::Identity();
        pose.T_wc.block<3, 3>(0, 0) = q.toRotationMatrix();
        pose.T_wc.block<3, 1>(0, 3) = Eigen::Vector3d(tx, ty, tz);

        trajectory.push_back(pose);
    }

    std::cout << "Loaded " << trajectory.size() << " poses from " << filepath << "\n";
    return trajectory;
}

PointCloud loadPointCloudXYZ(const std::string& filepath) {
    PointCloud cloud;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: could not open point cloud file: " << filepath << "\n";
        return cloud;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) {
            std::cerr << "Warning: skipping malformed line " << line_num
                      << " in " << filepath << "\n";
            continue;
        }

        Point3D point;
        point.position = Eigen::Vector3d(x, y, z);

        double r, g, b;
        if (iss >> r >> g >> b) {
            point.color = Eigen::Vector3f(
                static_cast<float>(r / 255.0),
                static_cast<float>(g / 255.0),
                static_cast<float>(b / 255.0));
        } else {
            point.color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
        }

        cloud.push_back(point);
    }

    std::cout << "Loaded " << cloud.size() << " points from " << filepath << "\n";
    return cloud;
}

} // namespace vio
