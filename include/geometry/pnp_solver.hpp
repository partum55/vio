#pragma once

#include "core/types.hpp"
#include "geometry/camera_model.hpp"

#include <Eigen/Dense>

#include <vector>

namespace vio {

    struct PnPParams {
        int min_points = 12;
        int iterations_count = 200;
        double reprojection_error = 4.0;
        double confidence = 0.995;
        double min_inlier_ratio = 0.35;
        double max_translation_update = 0.60;
    };

    struct PnPResult {
        bool success = false;
        FrameState pose;
        int inliers_count = 0;
    };

    class PnPSolver {
    public:
        explicit PnPSolver(
            const CameraIntrinsics& intrinsics,
            const PnPParams& params = PnPParams{}
        );

        PnPResult solve(
            const std::vector<Eigen::Vector3d>& points_3d_w,
            const std::vector<Eigen::Vector2d>& points_2d,
            const FrameState& initial_pose
        ) const;

    private:
        CameraIntrinsics intrinsics_;
        PnPParams params_;
    };

}
