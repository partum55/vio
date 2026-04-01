#pragma once

#include "core/dataset.h"
#include "core/types.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <filesystem>
#include <string>

namespace vio {

struct StreamSample {
    double timestamp_s = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    std::filesystem::path image_path;
    std::size_t track_count = 0;
};

class RerunStreamClient {
public:
    RerunStreamClient();
    ~RerunStreamClient();

    RerunStreamClient(const RerunStreamClient&) = delete;
    RerunStreamClient& operator=(const RerunStreamClient&) = delete;

    bool connect(const std::string& host, int port, int retries = 20, int retry_delay_ms = 250);
    bool sendInit(const Dataset& dataset);
    bool sendInit(const std::string& dataset_root,
                  int image_width,
                  int image_height,
                  double fx,
                  double fy,
                  double cx,
                  double cy,
                  double visual_scale = 1.0,
                  double ground_z = 0.0);
    bool sendSyntheticInit();
    bool sendPointCloud(const PointCloud& cloud, std::size_t max_points = 2500);
    bool sendSample(const StreamSample& sample);
    bool sendDone(std::size_t num_samples);
    bool isConnected() const;
    void close();

private:
    bool sendLine(const std::string& line);
    static std::string escapeJson(const std::string& value);
    static std::string vectorToJson(const Eigen::Vector3d& vec);
    static std::string quatToJson(const Eigen::Quaterniond& quat);

    int socket_fd_;
};

} // namespace vio
