#pragma once

#include <Eigen/Dense>

namespace vio {

struct Landmark {
    int id = -1;
    int track_id = -1;
    Eigen::Vector3d p_w = Eigen::Vector3d::Zero();
    bool valid = true;
    int num_observations = 0;
    double reprojection_error = 0.0;
};

} // namespace vio
