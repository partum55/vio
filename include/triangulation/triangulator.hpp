#pragma once

#include "core/tracked_frame.hpp"
#include "triangulation/camera_model.hpp"
#include "triangulation/landmark.hpp"

#include <vector>

struct TriangulationParams
{
    int min_observations = 2;
    double min_baseline = 0.05;
    double max_reprojection_error = 3.0;
    double min_depth = 1e-3;
};

std::vector<vio::Landmark> triangulateLandmarks(
    const std::vector<vio::TrackedFrame>& sequence,
    const CameraIntrinsics& camera_intrinsics,
    const TriangulationParams& params
);