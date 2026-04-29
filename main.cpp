#include "pipeline/vio_app_pipeline.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

    std::string joinPath(const std::string& a, const std::string& b)
    {
        if (a.empty()) {
            return b;
        }

        if (a.back() == '/') {
            return a + b;
        }

        return a + "/" + b;
    }

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            std::cerr << "Usage: ./vio <dataset_path>\n";
            return 1;
        }

        const std::string dataset_path = argv[1];

        std::cout << "Running NEW VIO pipeline\n";
        std::cout << "Dataset: " << dataset_path << "\n";

        CameraIntrinsics intrinsics;
        intrinsics.fx = 458.654;
        intrinsics.fy = 457.296;
        intrinsics.cx = 367.215;
        intrinsics.cy = 248.375;
        intrinsics.width = 752;
        intrinsics.height = 480;

        vio::VioAppPipeline pipeline(intrinsics);

        pipeline.setImuCsvPath(
            joinPath(dataset_path, "imu0/data.csv")
        );

        pipeline.setImagesDir(
            joinPath(dataset_path, "cam0/data")
        );

        pipeline.setFrameTimestampsPath(
            joinPath(dataset_path, "cam0/data.csv")
        );

        pipeline.setOutputPosesCsv("poses.csv");
        pipeline.setOutputObservationsCsv("observations.csv");
        pipeline.setOutputVideoPath("output.mp4");

        pipeline.setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));

        if (!pipeline.run()) {
            throw std::runtime_error("Pipeline failed");
        }

        std::cout << "Done.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}