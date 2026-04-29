#pragma once

#include "core/tracked_frame.hpp"
#include "geometry/camera_model.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

bool projectLandmarkToFrame(
    const vio::FrameState& state,
    const CameraIntrinsics& intrinsics,
    const Eigen::Vector3d& X_w,
    cv::Point2f& uv
);

bool projectLandmarkToFrame(
    const vio::FrameState& state,
    const CameraIntrinsics& intrinsics,
    const Eigen::Vector3d& X_w,
    Eigen::Vector2d& uv,
    double& depth
);
