#include "geometry/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <cmath>

namespace vio {

PnPSolver::PnPSolver(
    const CameraIntrinsics& intrinsics,
    const PnPParams& params
)
    : intrinsics_(intrinsics),
      params_(params)
{
}

PnPResult PnPSolver::solve(
    const std::vector<Eigen::Vector3d>& points_3d_w,
    const std::vector<Eigen::Vector2d>& points_2d,
    const FrameState& initial_pose
) const {
    PnPResult result;
    result.pose = initial_pose;

    if (!intrinsics_.isValid()) {
        return result;
    }

    if (points_3d_w.size() != points_2d.size()) {
        return result;
    }

    if (static_cast<int>(points_3d_w.size()) < params_.min_points) {
        return result;
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;

    object_points.reserve(points_3d_w.size());
    image_points.reserve(points_2d.size());

    for (std::size_t i = 0; i < points_3d_w.size(); ++i) {
        const Eigen::Vector3d& p3 = points_3d_w[i];
        const Eigen::Vector2d& p2 = points_2d[i];

        if (!p3.allFinite() || !p2.allFinite()) {
            continue;
        }

        object_points.emplace_back(
            static_cast<float>(p3.x()),
            static_cast<float>(p3.y()),
            static_cast<float>(p3.z())
        );

        image_points.emplace_back(
            static_cast<float>(p2.x()),
            static_cast<float>(p2.y())
        );
    }

    if (static_cast<int>(object_points.size()) < params_.min_points) {
        return result;
    }

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        intrinsics_.fx, 0.0, intrinsics_.cx,
        0.0, intrinsics_.fy, intrinsics_.cy,
        0.0, 0.0, 1.0
    );

    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;

    const bool ok = cv::solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        false,
        params_.iterations_count,
        static_cast<float>(params_.reprojection_error),
        params_.confidence,
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );

    const double inlier_ratio = object_points.empty()
        ? 0.0
        : static_cast<double>(inliers.rows) / static_cast<double>(object_points.size());

    if (!ok || inliers.rows < params_.min_points ||
        inlier_ratio < params_.min_inlier_ratio) {
        return result;
    }

    cv::Mat R_cw_cv;
    cv::Rodrigues(rvec, R_cw_cv);

    Eigen::Matrix3d R_cw;
    cv::cv2eigen(R_cw_cv, R_cw);

    Eigen::Vector3d t_cw(
        tvec.at<double>(0),
        tvec.at<double>(1),
        tvec.at<double>(2)
    );

    const Eigen::Matrix3d R_wc = R_cw.transpose();
    const Eigen::Vector3d t_wc = -R_wc * t_cw;

    if (!t_wc.allFinite()) {
        return result;
    }

    result.success = true;
    result.pose = initial_pose;
    result.pose.q_wc = Eigen::Quaterniond(R_wc).normalized();
    result.pose.t_wc = t_wc;
    result.inliers_count = inliers.rows;

    return result;
}

} // namespace vio
