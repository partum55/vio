#pragma once

#include "geometry/landmark.hpp"
#include <string>
#include <vector>

namespace vio {

bool writeLandmarksCsv(
    const std::string& path,
    const std::vector<Landmark>& landmarks
);

} // namespace vio
