#pragma once

#include "core/tracked_frame.hpp"
#include "imu/imu_processor.hpp"          // declarations only; src/imu/imu_processor.cpp may stay disabled in CMake for now
#include "tracking/tracking_vis.hpp"
#include "tracking/feature_refresh.hpp"
#include "geometry/camera_model.hpp"
#include "geometry/landmark.hpp"
#include "geometry/triangulator.hpp"
#include "geometry/extrinsics.hpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class ImuTrackingPipeline {
public:
    ImuTrackingPipeline() = default;

    void setImuCsvPath(const std::string& path);
    void setImagesDir(const std::string& path);
    void setFrameTimestampsPath(const std::string& path);

    void setOutputPosesCsv(const std::string& path);
    void setOutputObservationsCsv(const std::string& path);
    void setOutputVideoPath(const std::string& path);

    void setGravity(const Eigen::Vector3d& gravity);
    void setTrackingParams(int win_size, int max_level, int max_iters, float eps);
    void setRefreshParams(const FeatureRefreshParams& params);

    bool run();
    bool runOnlyTrackingAndSync();

    const std::vector<vio::TrackedFrame>& sequence() const;
    const std::vector<Pose>& imuTrajectory() const;

    void setCameraIntrinsics(const CameraIntrinsics& intrinsics);
    void setTriangulationParams(const TriangulationParams& params);

    const std::vector<vio::Landmark>& landmarks() const;

    void setOutputLandmarksCsv(const std::string& path);
    void setCameraExtrinsics(const RigidTransform& T_bs);

    bool runTrackingWithImuPrior();

private:
    bool loadInputs();
    bool runImu();
    bool runTrackingAndSync();
    void initializeTracks(const cv::Mat& first_gray);
    void appendFrame(int frame_id, double timestamp, const std::vector<Track>& tracks);
    bool runTriangulation();

    std::string imu_csv_path_;
    std::string images_dir_;
    std::string frame_timestamps_path_;

    std::string output_poses_csv_ = "poses.csv";
    std::string output_observations_csv_ = "observations.csv";
    std::string output_video_path_ = "imu_tracking_visualization.mp4";
    std::string output_landmarks_csv_ = "landmarks.csv";

    Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, 9.81);

    int win_size_ = 9;
    int max_level_ = 3;
    int max_iters_ = 10;
    float eps_ = 1e-3f;

    // Kept for future IMU integration. In the current visual-only mode they remain empty.
    std::vector<ImuSample> imu_samples_;
    std::vector<Pose> imu_trajectory_;

    std::vector<std::string> image_paths_;
    std::vector<double> frame_timestamps_;

    std::vector<vio::TrackedFrame> sequence_;
    CameraIntrinsics camera_intrinsics_;

    TriangulationParams triangulation_params_;
    std::vector<vio::Landmark> landmarks_;

    RigidTransform T_bs_;
};
