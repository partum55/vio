#include "pipeline/vio_app_pipeline.hpp"

#include <iostream>

namespace vio {

VioAppPipeline::VioAppPipeline(
    const CameraIntrinsics& intrinsics,
    const VioAppPipelineParams& params
)
    : intrinsics_(intrinsics),
      params_(params),
      input_pipeline_(),
      vio_pipeline_(
          GeometryBackend(intrinsics_, params_.geometry),
          params_.vio
      )
{
    input_pipeline_.setCameraIntrinsics(intrinsics_);
}

void VioAppPipeline::setImuCsvPath(const std::string& path) {
    input_pipeline_.setImuCsvPath(path);
}

void VioAppPipeline::setImagesDir(const std::string& path) {
    input_pipeline_.setImagesDir(path);
}

void VioAppPipeline::setFrameTimestampsPath(const std::string& path) {
    input_pipeline_.setFrameTimestampsPath(path);
}

void VioAppPipeline::setOutputPosesCsv(const std::string& path) {
    input_pipeline_.setOutputPosesCsv(path);
}

void VioAppPipeline::setOutputObservationsCsv(const std::string& path) {
    input_pipeline_.setOutputObservationsCsv(path);
}

void VioAppPipeline::setOutputVideoPath(const std::string& path) {
    input_pipeline_.setOutputVideoPath(path);
}

void VioAppPipeline::setGravity(const Eigen::Vector3d& gravity) {
    input_pipeline_.setGravity(gravity);
}

bool VioAppPipeline::run() {
    vio_pipeline_.reset();

    if (!input_pipeline_.runOnlyTrackingAndSync()) {
        std::cerr << "Input tracking pipeline failed\n";
        return false;
    }

    const std::vector<TrackedFrame>& tracked_sequence =
        input_pipeline_.sequence();

    if (tracked_sequence.empty()) {
        std::cerr << "Tracked sequence is empty\n";
        return false;
    }

    for (const TrackedFrame& frame : tracked_sequence) {
        vio_pipeline_.processFrame(frame);
    }

    std::cout << "VIO geometry pipeline finished\n";
    std::cout << "Frames: " << vio_pipeline_.frames().size() << "\n";
    std::cout << "Landmarks: " << vio_pipeline_.landmarks().size() << "\n";

    return true;
}

const std::vector<TrackedFrame>& VioAppPipeline::frames() const {
    return vio_pipeline_.frames();
}

const LandmarkMap& VioAppPipeline::landmarks() const {
    return vio_pipeline_.landmarks();
}

} // namespace vio