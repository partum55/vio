#include "core/projection.hpp"

#include <cmath>

bool projectLandmarkToFrame(
    const vio::FrameState& state,
    const CameraIntrinsics& intrinsics,
    const Eigen::Vector3d& X_w,
    Eigen::Vector2d& uv,
    double& depth
) {
    const Eigen::Matrix3d R_wc = state.q_wc.toRotationMatrix();
    const Eigen::Matrix3d R_cw = R_wc.transpose();
    const Eigen::Vector3d t_cw = -R_cw * state.t_wc;

    const Eigen::Vector3d X_c = R_cw * X_w + t_cw;
    depth = X_c.z();

    if (depth <= 0.0) {
        return false;
    }

    const double x = X_c.x() / depth;
    const double y = X_c.y() / depth;

    uv.x() = intrinsics.fx * x + intrinsics.cx;
    uv.y() = intrinsics.fy * y + intrinsics.cy;

    return std::isfinite(uv.x()) && std::isfinite(uv.y());
}

bool projectLandmarkToFrame(
    const vio::FrameState& state,
    const CameraIntrinsics& intrinsics,
    const Eigen::Vector3d& X_w,
    cv::Point2f& uv
) {
    Eigen::Vector2d uv_eigen;
    double depth = 0.0;

    const bool ok = projectLandmarkToFrame(
        state,
        intrinsics,
        X_w,
        uv_eigen,
        depth
    );

    if (!ok) {
        return false;
    }

    uv.x = static_cast<float>(uv_eigen.x());
    uv.y = static_cast<float>(uv_eigen.y());
    return true;
}