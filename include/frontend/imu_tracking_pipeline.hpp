#pragma once
#include "core/tracked_frame.hpp"
#include "imu/imu.hpp"
#include "tracking/tracking_vis.hpp"
#include "tracking/feature_refresh.hpp"

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
    void setTrackingParams(
        int win_size,
        int max_level,
        int max_iters,
        float eps
    );

    void setRefreshParams(const FeatureRefreshParams& params);

    bool run();

    const std::vector<vio::TrackedFrame>& sequence() const;
    const std::vector<Pose>& imuTrajectory() const;

private:
    bool loadInputs();
    bool runImu();
    bool runTrackingAndSync();
    void initializeTracks(const cv::Mat& first_gray);
    void appendFrame(
        int frame_id,
        double timestamp,
        const std::vector<Track>& tracks
    );

private:
    std::string imu_csv_path_;
    std::string images_dir_;
    std::string frame_timestamps_path_;

    std::string output_poses_csv_ = "poses.csv";
    std::string output_observations_csv_ = "observations.csv";
    std::string output_video_path_ = "imu_tracking_visualization.mp4";

    Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, 9.81);

    int win_size_ = 9;
    int max_level_ = 3;
    int max_iters_ = 10;
    float eps_ = 1e-3f;

    std::vector<ImuSample> imu_samples_;
    std::vector<Pose> imu_trajectory_;
    std::vector<std::string> image_paths_;
    std::vector<double> frame_timestamps_;

    std::vector<vio::TrackedFrame> sequence_;
};