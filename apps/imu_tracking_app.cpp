#include "frontend/imu_tracking_pipeline.hpp"

#include <Eigen/Dense>
#include <iostream>

int main()
{
    ImuTrackingPipeline pipeline;

    pipeline.setImuCsvPath("data/imu0/data.csv");
    pipeline.setImagesDir("data/cam0/undistorted_alpha0/");
    pipeline.setFrameTimestampsPath("data/frame_timestamps.txt");

    CameraIntrinsics intrinsics;
    intrinsics.fx = 356.3485;
    intrinsics.fy = 418.1912;
    intrinsics.cx = 363.0043;
    intrinsics.cy = 250.2713;

    pipeline.setCameraIntrinsics(intrinsics);

    pipeline.setOutputPosesCsv("results/poses.csv");
    pipeline.setOutputObservationsCsv("results/observations.csv");
    pipeline.setOutputVideoPath("results/imu_tracking_visualization.mp4");
    pipeline.setOutputLandmarksCsv("results/landmarks.csv");

    pipeline.setGravity(Eigen::Vector3d(0.0, 0.0, 9.81));
    pipeline.setTrackingParams(9, 3, 10, 1e-3f);

    if (!pipeline.run()) {
        std::cerr << "IMU + tracking pipeline failed\n";
        return 1;
    }
    return 0;
}