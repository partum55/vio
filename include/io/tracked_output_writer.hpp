#pragma once
#include "core/tracked_frame.hpp"
#include <string>
#include <vector>

bool writeFrameStatesCsv(
    const std::string& path,
    const std::vector<vio::TrackedFrame>& sequence
);

bool writeObservationsCsv(
    const std::string& path,
    const std::vector<vio::TrackedFrame>& sequence
);