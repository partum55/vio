#include "io/image_sequence_reader.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

namespace vio {

std::vector<std::string> loadImagePaths(
    const std::string& directory,
    const std::string& extension
)
{
    namespace fs = std::filesystem;

    std::vector<std::string> paths;

    if (!fs::exists(directory)) {
        std::cerr << "Image directory does not exist: " << directory << "\n";
        return paths;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        if (entry.path().extension() == extension) {
            paths.push_back(entry.path().string());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<double> loadImageTimestampsFromFilenames(
    const std::vector<std::string>& image_paths
)
{
    namespace fs = std::filesystem;

    std::vector<double> timestamps;
    timestamps.reserve(image_paths.size());

    for (const auto& path : image_paths) {
        fs::path p(path);
        const std::string stem = p.stem().string();

        try {
            const long long ts_ns = std::stoll(stem);
            const double ts_sec = static_cast<double>(ts_ns) * 1e-9;
            timestamps.push_back(ts_sec);
        } catch (const std::exception&) {
            std::cerr << "Failed to parse timestamp from filename: " << path << "\n";
            timestamps.clear();
            return timestamps;
        }
    }

    return timestamps;
}

std::vector<double> loadImageTimestampsFromFile(const std::string& path)
{
    std::vector<double> timestamps;

    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open frame timestamps file: " << path << "\n";
        return timestamps;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        try {
            const long long ts_ns = std::stoll(line);
            timestamps.push_back(static_cast<double>(ts_ns) * 1e-9);
        } catch (const std::exception&) {
            std::cerr << "Failed to parse timestamp line: " << line << "\n";
            timestamps.clear();
            return timestamps;
        }
    }

    return timestamps;
}

} // namespace vio
