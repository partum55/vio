#pragma once

#include "vio/types.h"

namespace vio {

struct GeneratorConfig {
    // Trajectory
    int num_poses = 300;
    double helix_radius = 4.0;
    double helix_height = 3.0;
    double helix_turns = 2.0;

    // Point cloud
    int num_points = 8000;
    double cloud_extent = 6.0;   // half-size of bounding box in XZ
    double cloud_height = 4.0;   // height range [0, cloud_height]
};

Trajectory generateTrajectory(const GeneratorConfig& config = {});
PointCloud generatePointCloud(const GeneratorConfig& config = {});

} // namespace vio
