#pragma once

#include "core/types.hpp"
#include "pipeline/vio_pipeline.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace vio {

class RerunVisualizer {
public:
    explicit RerunVisualizer(const std::string& app_name = "vio_pipeline");
    ~RerunVisualizer();

    RerunVisualizer(const RerunVisualizer&) = delete;
    RerunVisualizer& operator=(const RerunVisualizer&) = delete;

    bool connect(const std::string& host = "127.0.0.1", int port = 9876, int retries = 20, int retry_delay_ms = 250);
    bool isConnected() const noexcept;
    void close();

    void logPose(const FrameState& pose);
    void logTrajectory(const std::vector<FrameState>& trajectory);
    void logLandmarks(const std::vector<Landmark>& landmarks);
    void logTrackedFeatures(const std::vector<Observation>& features);
    void logStatus(VioStatus status);
    void logImagePath(const std::filesystem::path& image_path, double timestamp_s);

private:
    bool sendLine(const std::string& line);
    bool sendInit();
    static std::string escapeJson(const std::string& value);
    static std::string vector3ToJson(const Eigen::Vector3d& vec);
    static std::string vector2ToJson(const Eigen::Vector2d& vec);
    static std::string quatToJson(const Eigen::Quaterniond& quat);
    static const char* statusToString(VioStatus status);

    std::string app_name_;
    int socket_fd_ = -1;
};

} // namespace vio
