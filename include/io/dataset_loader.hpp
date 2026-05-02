#pragma once

#include "core/types.hpp"
#include "imu/imu_processor.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vio {

struct CameraCalibration {
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::string distortion_model;
    std::vector<double> distortion_coeffs;
    Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();
};

struct ImuCalibration {
    Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();
};

struct DatasetFrame {
    std::int64_t timestamp_ns = 0;
    double timestamp_s = 0.0;
    std::size_t frame_index = 0;
    std::filesystem::path image_path;
};

struct Dataset {
    std::filesystem::path root;
    CameraCalibration camera;
    ImuCalibration imu;
    std::vector<DatasetFrame> frames;
    std::vector<ImuSample> imu_samples;
};

struct DatasetLoadOptions {
    std::filesystem::path imu_csv_path;
    std::filesystem::path images_dir;
    std::filesystem::path frame_timestamps_path;
    CameraIntrinsics camera_intrinsics;
};

Dataset loadDataset(const DatasetLoadOptions& options);
Dataset loadEurocDataset(const std::filesystem::path& root);

} // namespace vio
