#pragma once

#include "core/types.h"

namespace vio {

struct GeneratorConfig {
    int num_poses = 300;
    double helix_radius = 4.0;
    double helix_height = 3.0;
    double helix_turns = 2.0;

    int num_points = 8000;
    double cloud_extent = 6.0;
    double cloud_height = 4.0;
};

Trajectory generateTrajectory(const GeneratorConfig& config = {});
PointCloud generatePointCloud(const GeneratorConfig& config = {});

} // namespace vio
