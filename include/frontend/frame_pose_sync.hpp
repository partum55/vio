#pragma once
#include "core/types.hpp"
#include "imu/imu.hpp"
#include <vector>

size_t findClosestPoseIndex(
    double timestamp,
    const std::vector<Pose>& imu_trajectory
);

vio::FrameState buildFrameStateFromImu(
    int frame_id,
    double frame_timestamp,
    const std::vector<Pose>& imu_trajectory
);