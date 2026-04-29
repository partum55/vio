#pragma once

#include <Eigen/Dense>

namespace vio {

struct RigidTransform
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

} // namespace vio
