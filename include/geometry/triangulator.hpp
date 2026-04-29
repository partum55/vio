#pragma once

#include "core/tracked_frame.hpp"
#include "geometry/camera_model.hpp"
#include "geometry/landmark.hpp"

#include <vector>

struct TriangulationParams {
    int min_observations = 2;
    double min_baseline = 0.06;
    double max_baseline = 0.90;
    double max_reprojection_error = 4.0;
    double min_depth = 0.05;
};

std::vector<vio::Landmark> triangulateLandmarks(
    const std::vector<vio::TrackedFrame>& sequence,
    const CameraIntrinsics& camera_intrinsics,
    const TriangulationParams& params
);
