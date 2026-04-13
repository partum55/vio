#include "frontend/imu_tracking_pipeline.hpp"

#include <Eigen/Dense>
#include <iostream>

int main()
{
    ImuTrackingPipeline pipeline;

    pipeline.setImuCsvPath("../imu0/data.csv");
    pipeline.setImuCsvPath("data/imu/data.csv");
    pipeline.setImagesDir("data/cam0/undistorted_alpha0");

    pipeline.setOutputPosesCsv("poses.csv");
    pipeline.setOutputObservationsCsv("observations.csv");
    pipeline.setOutputVideoPath("imu_tracking_visualization.mp4");

    pipeline.setGravity(Eigen::Vector3d(0.0, 0.0, 9.81));
    pipeline.setTrackingParams(9, 3, 10, 1e-3f);

    if (!pipeline.run()) {
        std::cerr << "IMU + tracking pipeline failed\n";
        return 1;
    }
    return 0;
}