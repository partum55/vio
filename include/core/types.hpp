#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace vio {

struct FrameState {
    int frame_id = -1;
    double timestamp = 0.0;
    Eigen::Quaterniond q_wc = Eigen::Quaterniond(1, 0, 0, 0);
    Eigen::Vector3d t_wc = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_w = Eigen::Vector3d::Zero();
};

struct Observation {
    int frame_id = -1;
    int track_id = -1;
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    bool valid = true;
};

} 