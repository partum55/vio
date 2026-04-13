#pragma once
#include <string>
#include <vector>

std::vector<std::string> loadImagePaths(
    const std::string& directory,
    const std::string& extension = ".png"
);

std::vector<double> loadImageTimestampsFromFilenames(
    const std::vector<std::string>& image_paths
);