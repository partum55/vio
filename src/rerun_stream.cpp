#include "vio/rerun_stream.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <thread>

namespace vio {

RerunStreamClient::RerunStreamClient()
    : socket_fd_(-1) {}

RerunStreamClient::~RerunStreamClient() {
    close();
}

bool RerunStreamClient::connect(const std::string& host, int port, int retries, int retry_delay_ms) {
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
            return true;
        }

        close();
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
    }

    return false;
}

bool RerunStreamClient::sendInit(const std::string& scene_name) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"init\","
         << "\"dataset_root\":\"" << escapeJson(scene_name) << "\","
         << "\"image_width\":1280,"
         << "\"image_height\":720,"
         << "\"fx\":700.0,"
         << "\"fy\":700.0,"
         << "\"cx\":640.0,"
         << "\"cy\":360.0"
         << "}";
    return sendLine(json.str());
}

bool RerunStreamClient::sendPointCloud(const PointCloud& cloud, std::size_t max_points) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"cloud\","
         << "\"positions\":[";

    const std::size_t limit = std::min(max_points, cloud.size());
    for (std::size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            json << ",";
        }
        json << vectorToJson(cloud[i].position);
    }

    json << "],\"colors\":[";
    for (std::size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            json << ",";
        }
        json << "["
             << static_cast<int>(std::lround(cloud[i].color.x() * 255.0f)) << ","
             << static_cast<int>(std::lround(cloud[i].color.y() * 255.0f)) << ","
             << static_cast<int>(std::lround(cloud[i].color.z() * 255.0f)) << "]";
    }
    json << "]}";
    return sendLine(json.str());
}

bool RerunStreamClient::sendSample(const StreamSample& sample) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"sample\","
         << "\"timestamp\":" << sample.timestamp_s << ","
         << "\"position\":" << vectorToJson(sample.position) << ","
         << "\"orientation_xyzw\":" << quatToJson(sample.orientation) << ","
         << "\"gyro\":" << vectorToJson(sample.gyro) << ","
         << "\"acc\":" << vectorToJson(sample.acc) << ","
         << "\"track_count\":" << sample.track_count << ","
         << "\"image_path\":\"" << escapeJson(sample.image_path.string()) << "\""
         << "}";
    return sendLine(json.str());
}

bool RerunStreamClient::sendDone(std::size_t num_samples) {
    std::ostringstream json;
    json << "{"
         << "\"type\":\"done\","
         << "\"num_samples\":" << num_samples
         << "}";
    return sendLine(json.str());
}

bool RerunStreamClient::isConnected() const {
    return socket_fd_ >= 0;
}

void RerunStreamClient::close() {
    if (socket_fd_ >= 0) {
        ::shutdown(socket_fd_, SHUT_RDWR);
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

bool RerunStreamClient::sendLine(const std::string& line) {
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

std::string RerunStreamClient::escapeJson(const std::string& value) {
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

std::string RerunStreamClient::vectorToJson(const Eigen::Vector3d& vec) {
    std::ostringstream json;
    json << "[" << vec.x() << "," << vec.y() << "," << vec.z() << "]";
    return json.str();
}

std::string RerunStreamClient::quatToJson(const Eigen::Quaterniond& quat) {
    std::ostringstream json;
    json << "[" << quat.x() << "," << quat.y() << "," << quat.z() << "," << quat.w() << "]";
    return json.str();
}

} // namespace vio
