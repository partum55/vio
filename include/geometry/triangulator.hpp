#pragma once

#include "core/types.hpp"

#include <vector>

namespace vio {

struct TriangulationParams {
    int min_observations = 2;
    double min_baseline = 0.06;
    double max_baseline = 0.90;
    double max_reprojection_error = 4.0;
    double min_depth = 0.05;
};

std::vector<Landmark> triangulateLandmarks(
    const std::vector<TrackedFrame>& sequence,
    const CameraIntrinsics& camera_intrinsics,
    const TriangulationParams& params
);

} // namespace vio
