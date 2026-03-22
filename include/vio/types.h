#pragma once

#include <Eigen/Core>
#include <vector>

namespace vio {

struct Point3D {
    Eigen::Vector3d position;
    Eigen::Vector3f color; // RGB [0,1]
};

struct CameraPose {
    double timestamp;
    Eigen::Matrix4d T_wc; // world-from-camera SE3
};

using Trajectory = std::vector<CameraPose>;
using PointCloud = std::vector<Point3D>;

} // namespace vio
