#pragma once

#include "core/types.hpp"
#include "imu/imu_processor.hpp"

#include <vector>
#include <cstddef>

std::size_t findClosestPoseIndex(
    double timestamp,
    const std::vector<Pose>& imu_trajectory
);

vio::FrameState buildFrameStateFromImu(
    int frame_id,
    double frame_timestamp,
    const std::vector<Pose>& imu_trajectory
);
