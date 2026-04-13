#pragma once

#include "triangulation/landmark.hpp"

#include <string>
#include <vector>

bool writeLandmarksCsv(
    const std::string& path,
    const std::vector<vio::Landmark>& landmarks
);