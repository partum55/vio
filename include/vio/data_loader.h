#pragma once

#include "vio/types.h"
#include <string>

namespace vio {

/// Load trajectory from TUM format file (timestamp tx ty tz qx qy qz qw).
Trajectory loadTrajectoryTUM(const std::string& filepath);

/// Load point cloud from XYZ text file (x y z [r g b]).
PointCloud loadPointCloudXYZ(const std::string& filepath);

} // namespace vio
