#pragma once

#include "vio/types.h"
#include <string>

namespace vio {

struct RecordConfig {
    std::string output_path = "output.mp4";
    int width = 1920;
    int height = 1080;
    int fps = 60;
    double camera_distance = 3.0;   // chase camera distance behind agent
    double camera_height = 1.5;     // chase camera height above agent
    double smoothing = 0.05;        // camera smoothing factor (0=instant, 1=no move)
    int poses_per_frame = 1;        // how many trajectory steps per video frame
};

class Viewer {
public:
    Viewer(const Trajectory& trajectory, const PointCloud& cloud,
           const LineSet& lines = {});
    void run();                                  // interactive window (blocking)
    void record(const RecordConfig& cfg = {});   // render to video file (blocking)

private:
    void drawScene();
    void drawPointCloud();
    void drawTrajectory(int up_to_pose = -1);
    void drawCurrentFrustum(int pose_idx);
    void drawCameraFrustums();
    void drawDashedLines();
    void drawGrid();

    const Trajectory& trajectory_;
    const PointCloud& cloud_;
    const LineSet lines_;

    static constexpr int keyframe_step_ = 15;
    static constexpr float frustum_scale_ = 0.15f;
};

} // namespace vio
