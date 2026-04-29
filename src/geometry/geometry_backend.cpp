#include "geometry/geometry_backend.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vio {

namespace {

double rotationAngleDeg(
    const Eigen::Quaterniond& q1,
    const Eigen::Quaterniond& q2
) {
    Eigen::Quaterniond dq = q1.inverse() * q2;
    dq.normalize();

    const double w = std::clamp(dq.w(), -1.0, 1.0);
    const double angle_rad = 2.0 * std::acos(std::abs(w));

    constexpr double pi = 3.14159265358979323846;
    return angle_rad * 180.0 / pi;
}

}

GeometryBackend::GeometryBackend(
    const CameraIntrinsics& intrinsics,
    const GeometryBackendParams& params
)
    : intrinsics_(intrinsics),
      params_(params),
      pnp_solver_(intrinsics, params.pnp)
{
}

bool GeometryBackend::baselineEnough(
    const FrameState& pivot,
    const FrameState& current
) const {
    const double translation =
        (current.t_wc - pivot.t_wc).norm();

    const double rotation_deg =
        rotationAngleDeg(pivot.q_wc, current.q_wc);

    return translation >= params_.min_baseline_translation ||
           rotation_deg >= params_.min_baseline_rotation_deg;
}

GeometryStepResult GeometryBackend::triangulateTwoViews(
    const TrackedFrame& pivot,
    const TrackedFrame& current,
    LandmarkMap& landmark_map
) const {
    GeometryStepResult result;
    result.refined_pose = current.state;

    std::vector<TrackedFrame> sequence;
    sequence.reserve(2);
    sequence.push_back(pivot);
    sequence.push_back(current);

    const std::vector<Landmark> new_landmarks =
        triangulateLandmarks(
            sequence,
            intrinsics_,
            params_.triangulation
        );

    int created = 0;

    for (const Landmark& landmark : new_landmarks) {
        if (!landmark.valid) {
            continue;
        }

        landmark_map.addOrUpdate(
            landmark.track_id,
            landmark.p_w,
            landmark.reprojection_error,
            landmark.num_observations
        );

        ++created;
    }

    result.success = created > 0;
    result.created_landmarks = created;

    return result;
}

GeometryStepResult GeometryBackend::solvePnP(
    const TrackedFrame& current,
    const LandmarkMap& landmark_map
) const {
    GeometryStepResult result;
    result.refined_pose = current.state;

    std::vector<Eigen::Vector3d> points_3d_w;
    std::vector<Eigen::Vector2d> points_2d;

    landmark_map.buildPnPCorrespondences(
        current.observations,
        points_3d_w,
        points_2d
    );

    const PnPResult pnp_result =
        pnp_solver_.solve(
            points_3d_w,
            points_2d,
            current.state
        );

    if (!pnp_result.success) {
        return result;
    }

    result.success = true;
    result.refined_pose = pnp_result.pose;
    result.pnp_inliers = pnp_result.inliers_count;

    return result;
}

}