#pragma once

#include "core/types.hpp"

#include <string>
#include <vector>

namespace vio {

bool writeFrameStatesCsv(
    const std::string& path,
    const std::vector<TrackedFrame>& sequence
);

bool writeObservationsCsv(
    const std::string& path,
    const std::vector<TrackedFrame>& sequence
);

} // namespace vio
