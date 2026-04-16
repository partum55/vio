#pragma once
#include <string>
#include <vector>

std::vector<std::string> loadImagePaths(
    const std::string& directory,
    const std::string& extension
);

std::vector<double> loadImageTimestampsFromFilenames(
    const std::vector<std::string>& image_paths
);

std::vector<double> loadImageTimestampsFromFile(
    const std::string& path
);