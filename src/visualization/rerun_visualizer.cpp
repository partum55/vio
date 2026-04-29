#include "visualization/rerun_visualizer.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <sstream>
#include <thread>

namespace vio {

RerunVisualizer::RerunVisualizer(const std::string& app_name)
    : app_name_(app_name) {}

RerunVisualizer::~RerunVisualizer() {
    close();
}

bool RerunVisualizer::connect(const std::string& host, int port, int retries, int retry_delay_ms) {
    close();

    for (int attempt = 0; attempt < retries; ++attempt) {
        socket_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            return false;
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
            close();
            return false;
        }

        if (::connect(socket_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
            return sendInit();
        }

        close();
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
    }

    return false;
}

bool RerunVisualizer::isConnected() const noexcept {
    return socket_fd_ >= 0;
}

void RerunVisualizer::close() {
    if (socket_fd_ >= 0) {
        ::shutdown(socket_fd_, SHUT_RDWR);
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

void RerunVisualizer::logPose(const FrameState& pose) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"pose\","
         << "\"app\":\"" << escapeJson(app_name_) << "\","
         << "\"frame_id\":" << pose.frame_id << ","
         << "\"timestamp\":" << pose.timestamp << ","
         << "\"position\":" << vector3ToJson(pose.t_wc) << ","
         << "\"orientation_xyzw\":" << quatToJson(pose.q_wc) << ","
         << "\"velocity\":" << vector3ToJson(pose.v_w)
         << "}";
    (void)sendLine(json.str());
}

void RerunVisualizer::logTrajectory(const std::vector<FrameState>& trajectory) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"trajectory\","
         << "\"app\":\"" << escapeJson(app_name_) << "\","
         << "\"positions\":[";

    for (std::size_t i = 0; i < trajectory.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << vector3ToJson(trajectory[i].t_wc);
    }

    json << "]}";
    (void)sendLine(json.str());
}

void RerunVisualizer::logLandmarks(const std::vector<Landmark>& landmarks) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"landmarks\","
         << "\"app\":\"" << escapeJson(app_name_) << "\","
         << "\"points\":[";

    bool first = true;
    for (const Landmark& landmark : landmarks) {
        if (!landmark.valid) {
            continue;
        }
        if (!first) {
            json << ",";
        }
        first = false;
        json << "{"
             << "\"id\":" << landmark.id << ","
             << "\"track_id\":" << landmark.track_id << ","
             << "\"position\":" << vector3ToJson(landmark.p_w) << ","
             << "\"reprojection_error\":" << landmark.reprojection_error << ","
             << "\"num_observations\":" << landmark.num_observations
             << "}";
    }

    json << "]}";
    (void)sendLine(json.str());
}

void RerunVisualizer::logTrackedFeatures(const std::vector<Observation>& features) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"tracked_features\","
         << "\"app\":\"" << escapeJson(app_name_) << "\","
         << "\"features\":[";

    bool first = true;
    for (const Observation& observation : features) {
        if (!observation.valid) {
            continue;
        }
        if (!first) {
            json << ",";
        }
        first = false;
        json << "{"
             << "\"frame_id\":" << observation.frame_id << ","
             << "\"track_id\":" << observation.track_id << ","
             << "\"uv\":" << vector2ToJson(observation.uv)
             << "}";
    }

    json << "]}";
    (void)sendLine(json.str());
}

void RerunVisualizer::logStatus(VioStatus status) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"status\","
         << "\"app\":\"" << escapeJson(app_name_) << "\","
         << "\"status\":\"" << statusToString(status) << "\""
         << "}";
    (void)sendLine(json.str());
}

void RerunVisualizer::logImagePath(const std::filesystem::path& image_path, double timestamp_s) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"image\","
         << "\"app\":\"" << escapeJson(app_name_) << "\","
         << "\"timestamp\":" << timestamp_s << ","
         << "\"image_path\":\"" << escapeJson(image_path.string()) << "\""
         << "}";
    (void)sendLine(json.str());
}

bool RerunVisualizer::sendLine(const std::string& line) {
    if (socket_fd_ < 0) {
        return false;
    }

    const std::string payload = line + "\n";
    const char* data = payload.data();
    std::size_t total_sent = 0;

    while (total_sent < payload.size()) {
        const ssize_t sent = ::send(socket_fd_, data + total_sent, payload.size() - total_sent, 0);
        if (sent <= 0) {
            close();
            return false;
        }
        total_sent += static_cast<std::size_t>(sent);
    }

    return true;
}

bool RerunVisualizer::sendInit() {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"init\","
         << "\"app\":\"" << escapeJson(app_name_) << "\""
         << "}";
    return sendLine(json.str());
}

std::string RerunVisualizer::escapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped.push_back(ch);
            break;
        }
    }
    return escaped;
}

std::string RerunVisualizer::vector3ToJson(const Eigen::Vector3d& vec) {
    std::ostringstream json;
    json << "[" << vec.x() << "," << vec.y() << "," << vec.z() << "]";
    return json.str();
}

std::string RerunVisualizer::vector2ToJson(const Eigen::Vector2d& vec) {
    std::ostringstream json;
    json << "[" << vec.x() << "," << vec.y() << "]";
    return json.str();
}

std::string RerunVisualizer::quatToJson(const Eigen::Quaterniond& quat) {
    std::ostringstream json;
    json << "[" << quat.x() << "," << quat.y() << "," << quat.z() << "," << quat.w() << "]";
    return json.str();
}

const char* RerunVisualizer::statusToString(VioStatus status) {
    switch (status) {
    case VioStatus::Uninitialized:
        return "Uninitialized";
    case VioStatus::NeedFirstFrame:
        return "NeedFirstFrame";
    case VioStatus::TrackingFromPivot:
        return "TrackingFromPivot";
    case VioStatus::NeedInitialLandmarks:
        return "NeedInitialLandmarks";
    case VioStatus::TrackingWithMap:
        return "TrackingWithMap";
    case VioStatus::LostTracking:
        return "LostTracking";
    case VioStatus::Finished:
        return "Finished";
    case VioStatus::Failed:
        return "Failed";
    }
    return "Unknown";
}

} // namespace vio
