#pragma once

#include "core/types.hpp"
#include "geometry/landmark_map.hpp"
#include "geometry/pnp_solver.hpp"
#include "geometry/camera_model.hpp"
#include "geometry/triangulator.hpp"

namespace vio {

    struct GeometryBackendParams {
        TriangulationParams triangulation;
        PnPParams pnp;

        double min_baseline_translation = 0.05;
        double min_baseline_rotation_deg = 3.0;
    };

    struct GeometryStepResult {
        bool success = false;
        int created_landmarks = 0;
        int pnp_inliers = 0;
        FrameState refined_pose;
    };

    class GeometryBackend {
    public:
        explicit GeometryBackend(
            const CameraIntrinsics& intrinsics,
            const GeometryBackendParams& params = GeometryBackendParams{}
        );

        bool baselineEnough(
            const FrameState& pivot,
            const FrameState& current
        ) const;

        GeometryStepResult triangulateTwoViews(
            const TrackedFrame& pivot,
            const TrackedFrame& current,
            LandmarkMap& landmark_map
        ) const;

        GeometryStepResult solvePnP(
            const TrackedFrame& current,
            const LandmarkMap& landmark_map
        ) const;

    private:
        CameraIntrinsics intrinsics_;
        GeometryBackendParams params_;
        PnPSolver pnp_solver_;
    };

}
