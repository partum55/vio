#include "io/image_sequence_reader.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <cctype>

namespace {
    std::string trim(const std::string& s)
    {
        size_t begin = 0;
        while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
            ++begin;
        }

        size_t end = s.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
            --end;
        }

        return s.substr(begin, end - begin);
    }

    bool parseTimestampNsFromLine(const std::string& line, long long& timestamp_ns)
    {
        std::string cleaned = trim(line);
        if (cleaned.empty() || cleaned[0] == '#') {
            return false;
        }

        const size_t comma = cleaned.find(',');
        if (comma != std::string::npos) {
            cleaned = cleaned.substr(0, comma);
        }

        cleaned = trim(cleaned);
        if (cleaned.empty()) {
            return false;
        }

        try {
            size_t used = 0;
            timestamp_ns = std::stoll(cleaned, &used);
            return used == cleaned.size();
        } catch (const std::exception&) {
            return false;
        }
    }
}

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
            timestamps.push_back(static_cast<double>(ts_ns) * 1e-9);
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
        std::cerr << "Frame timestamps file is not available: " << path << "\n";
        return timestamps;
    }

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;

        long long ts_ns = 0;
        if (!parseTimestampNsFromLine(line, ts_ns)) {
            const std::string cleaned = trim(line);
            if (!cleaned.empty() && cleaned[0] != '#') {
                std::cerr << "Skipping invalid timestamp line " << line_no
                          << " in " << path << ": " << line << "\n";
            }
            continue;
        }

        timestamps.push_back(static_cast<double>(ts_ns) * 1e-9);
    }

    return timestamps;
}
