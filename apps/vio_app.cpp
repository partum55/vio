#include "pipeline/vio_pipeline.hpp"
#include "visualization/rerun_visualizer.hpp"

#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <vector>

int main()
{
    vio::VioRunConfig config;
    config.imu_csv_path = "../data/imu0/data.csv";
    config.images_dir = "../data/cam0/undistorted_alpha0/";
    config.frame_timestamps_path = "../data/frame_timestamps.txt";

    vio::CameraIntrinsics intrinsics;
    intrinsics.fx = 356.3485;
    intrinsics.fy = 418.1912;
    intrinsics.cx = 363.0043;
    intrinsics.cy = 250.2713;

    config.camera_intrinsics = intrinsics;

    config.output_poses_csv = "../results/poses.csv";
    config.output_observations_csv = "../results/observations.csv";
    config.output_video_path = "../results/imu_tracking_visualization.mp4";
    config.output_landmarks_csv = "../results/landmarks.csv";

    config.gravity = Eigen::Vector3d(0.0, 0.0, 9.81);
    config.tracker_win_size = 9;
    config.tracker_max_level = 3;
    config.tracker_max_iters = 10;
    config.tracker_eps = 1e-3f;

#if defined(VIO_ENABLE_RERUN_VISUALIZATION)
    vio::RerunVisualizer visualizer("vio_pipeline");
    (void)visualizer.connect();
    std::vector<vio::FrameState> trajectory;
    config.frame_logger =
        [&visualizer, &trajectory](
            const vio::TrackedFrame& frame,
            const std::vector<vio::Landmark>& landmarks,
            vio::VioStatus status,
            const std::filesystem::path& image_path) {
            trajectory.push_back(frame.state);
            visualizer.logPose(frame.state);
            visualizer.logTrajectory(trajectory);
            visualizer.logLandmarks(landmarks);
            visualizer.logTrackedFeatures(frame.observations);
            visualizer.logImagePath(image_path, frame.state.timestamp);
            visualizer.logStatus(status);
        };
#endif

    const vio::VioRunResult result = vio::VioPipeline::runConfigured(config);
    if (!result.success) {
        std::cerr << "VIO pipeline failed";
        if (!result.error.empty()) {
            std::cerr << ": " << result.error;
        }
        std::cerr << "\n";
        return 1;
    }

    std::cout << "VIO pipeline finished: "
              << result.frame_count << " frames, "
              << result.landmark_count << " landmarks\n";
    return 0;
}
