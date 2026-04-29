#include "frontend/frame_pose_sync.hpp"

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace vio {

namespace
{
    size_t findRightPoseIndex(
        double timestamp,
        const std::vector<Pose>& imu_trajectory
    )
    {
        auto it = std::lower_bound(
            imu_trajectory.begin(),
            imu_trajectory.end(),
            timestamp,
            [](const Pose& pose, double t) {
                return pose.t < t;
            }
        );

        return static_cast<size_t>(std::distance(imu_trajectory.begin(), it));
    }

    Pose interpolatePose(
        double timestamp,
        const std::vector<Pose>& imu_trajectory,
        size_t& left_idx,
        size_t& right_idx
    )
    {
        if (imu_trajectory.empty()) {
            throw std::runtime_error("interpolatePose: empty IMU trajectory");
        }

        if (timestamp <= imu_trajectory.front().t) {
            left_idx = 0;
            right_idx = 0;
            return imu_trajectory.front();
        }

        if (timestamp >= imu_trajectory.back().t) {
            left_idx = imu_trajectory.size() - 1;
            right_idx = imu_trajectory.size() - 1;
            return imu_trajectory.back();
        }

        right_idx = findRightPoseIndex(timestamp, imu_trajectory);

        if (right_idx == 0) {
            left_idx = 0;
            return imu_trajectory.front();
        }

        if (right_idx >= imu_trajectory.size()) {
            left_idx = imu_trajectory.size() - 1;
            right_idx = imu_trajectory.size() - 1;
            return imu_trajectory.back();
        }

        left_idx = right_idx - 1;

        const Pose& p0 = imu_trajectory[left_idx];
        const Pose& p1 = imu_trajectory[right_idx];

        const double dt = p1.t - p0.t;
        if (dt <= 1e-12) {
            return p0;
        }

        double alpha = (timestamp - p0.t) / dt;
        alpha = std::clamp(alpha, 0.0, 1.0);

        Pose out;
        out.t = timestamp;
        out.p = (1.0 - alpha) * p0.p + alpha * p1.p;
        out.v = (1.0 - alpha) * p0.v + alpha * p1.v;
        out.q = p0.q.slerp(alpha, p1.q);
        out.q.normalize();

        return out;
    }
}

FrameState buildFrameStateFromImu(
    int frame_id,
    double frame_timestamp,
    const std::vector<Pose>& imu_trajectory
)
{
    if (imu_trajectory.empty()) {
        throw std::runtime_error("buildFrameStateFromImu: empty IMU trajectory");
    }

    size_t left_idx = 0;
    size_t right_idx = 0;
    const Pose pose = interpolatePose(
        frame_timestamp,
        imu_trajectory,
        left_idx,
        right_idx
    );

    static int dbg_counter = 0;
    ++dbg_counter;

    FrameState state;
    state.frame_id = frame_id;
    state.timestamp = frame_timestamp;


    state.q_wc = pose.q.conjugate();
    state.q_wc.normalize();

    state.t_wc = pose.p;
    state.v_w = pose.v;

    return state;
}

} // namespace vio
