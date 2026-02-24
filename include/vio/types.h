#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <limits>
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

struct Line3D {
    Eigen::Vector3d start;
    Eigen::Vector3d end;
    Eigen::Vector3f color; // RGB [0,1]
};

struct Joint2D {
    int id = -1;
    cv::Point2f uv = cv::Point2f(0.0f, 0.0f);
    float confidence = 1.0f;
};

struct Pose2D {
    double timestamp = 0.0;
    std::vector<Joint2D> joints;
};

struct Joint3D {
    int id = -1;
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
    float confidence = 0.0f;
    bool valid = false;
    double reproj_err_px = std::numeric_limits<double>::infinity();
};

struct Pose3D {
    double timestamp = 0.0;
    std::vector<Joint3D> joints;
};

using Trajectory = std::vector<CameraPose>;
using PointCloud = std::vector<Point3D>;
using LineSet = std::vector<Line3D>;
using Pose2DSequence = std::vector<Pose2D>;
using Pose3DSequence = std::vector<Pose3D>;

} // namespace vio
