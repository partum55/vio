#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>

namespace vio {

struct Point3D {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3f color = Eigen::Vector3f::Zero(); // RGB [0,1]
};

struct CameraPose {
    double timestamp = 0.0;
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity(); // world-from-camera SE3
};

struct FrameState {
    int frame_id = -1;
    double timestamp = 0.0;
    Eigen::Quaterniond q_wc = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wc = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_w = Eigen::Vector3d::Zero();
};

struct Observation {
    int frame_id = -1;
    int track_id = -1;
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    bool valid = true;
};

using Trajectory = std::vector<CameraPose>;
using PointCloud = std::vector<Point3D>;

} // namespace vio
