#pragma once

#include "core/types.h"

#include <string>

namespace vio {

struct RecordConfig {
    std::string output_path = "output.mp4";
    int width = 1920;
    int height = 1080;
    int fps = 60;
    double camera_distance = 3.0;
    double camera_height = 1.5;
    double smoothing = 0.05;
    int poses_per_frame = 1;
};

class Viewer {
public:
    Viewer(const Trajectory& trajectory, const PointCloud& cloud);
    void run();
    void record(const RecordConfig& cfg = {});

private:
    void drawScene();
    void drawPointCloud();
    void drawTrajectory(int up_to_pose = -1);
    void drawCurrentFrustum(int pose_idx);
    void drawCameraFrustums();
    void drawGrid();

    const Trajectory& trajectory_;
    const PointCloud& cloud_;

    static constexpr int keyframe_step_ = 15;
    static constexpr float frustum_scale_ = 0.15f;
};

} // namespace vio
