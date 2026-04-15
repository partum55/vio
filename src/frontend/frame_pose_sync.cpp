#include "frontend/frame_pose_sync.hpp"

#include <stdexcept>
#include <cmath>

size_t findClosestPoseIndex(
    double timestamp,
    const std::vector<Pose>& imu_trajectory
)
{
    if (imu_trajectory.empty()) {
        throw std::runtime_error("findClosestPoseIndex: empty IMU trajectory");
    }

    size_t best_idx = 0;
    double best_diff = std::abs(imu_trajectory[0].t - timestamp);

    for (size_t i = 1; i < imu_trajectory.size(); ++i) {
        const double diff = std::abs(imu_trajectory[i].t - timestamp);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }

    return best_idx;
}

vio::FrameState buildFrameStateFromImu(
    int frame_id,
    double frame_timestamp,
    const std::vector<Pose>& imu_trajectory
)
{
    if (imu_trajectory.empty()) {
        throw std::runtime_error("buildFrameStateFromImu: empty IMU trajectory");
    }

    const size_t idx = findClosestPoseIndex(frame_timestamp, imu_trajectory);
    const Pose& pose = imu_trajectory[idx];

    vio::FrameState state;
    state.frame_id = frame_id;
    state.timestamp = frame_timestamp;
    state.q_wc = pose.q;
    state.t_wc = pose.p;
    state.v_w = pose.v;

    return state;
}