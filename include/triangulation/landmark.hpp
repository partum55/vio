#pragma once

#include <Eigen/Dense>

namespace vio
{
    struct Landmark
    {
        int track_id = -1;
        Eigen::Vector3d p_w = Eigen::Vector3d::Zero();
        bool valid = false;
        double reprojection_error = 0.0;
        int num_observations = 0;
    };
}