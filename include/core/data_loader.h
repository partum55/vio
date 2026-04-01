#pragma once

#include "core/types.h"

#include <string>

namespace vio {

Trajectory loadTrajectoryTUM(const std::string& filepath);
PointCloud loadPointCloudXYZ(const std::string& filepath);

} // namespace vio
