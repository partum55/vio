#pragma once

#include "frontend/imu_tracking_pipeline.hpp"
#include "geometry/geometry_backend.hpp"
#include "geometry/landmark_map.hpp"
#include "pipeline/vio_pipeline.hpp"
#include "geometry/camera_model.hpp"

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace vio {

    struct VioAppPipelineParams {
        VioPipelineParams vio;
        GeometryBackendParams geometry;
    };

    class VioAppPipeline {
    public:
        explicit VioAppPipeline(
            const CameraIntrinsics& intrinsics,
            const VioAppPipelineParams& params = VioAppPipelineParams{}
        );

        void setImuCsvPath(const std::string& path);
        void setImagesDir(const std::string& path);
        void setFrameTimestampsPath(const std::string& path);

        void setOutputPosesCsv(const std::string& path);
        void setOutputObservationsCsv(const std::string& path);
        void setOutputVideoPath(const std::string& path);

        void setGravity(const Eigen::Vector3d& gravity);

        bool run();

        const std::vector<TrackedFrame>& frames() const;
        const LandmarkMap& landmarks() const;

    private:
        CameraIntrinsics intrinsics_;
        VioAppPipelineParams params_;

        ::ImuTrackingPipeline input_pipeline_;
        VioPipeline vio_pipeline_;
    };

} // namespace vio