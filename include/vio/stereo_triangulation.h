#pragma once

#include "vio/types.h"

#include <opencv2/core.hpp>

#include <stdexcept>
#include <string>

namespace vio {

struct TriangulationConfig {
    int max_features = 5000;
    float match_ratio_thresh = 0.75f;
    int min_matches = 20;
    double max_reproj_error = 3.0;
    double max_depth = 100.0;
    bool show_matches = true;   // show 2D match visualization window

    // Joint-based pose triangulation thresholds.
    double min_joint_confidence = 0.35;
    double max_epipolar_error_px = 2.0;
    double min_disparity_px = 1.5;
    double max_reprojection_error_px = 2.0;

    // Temporal smoothing for sequence mode.
    double temporal_smoothing_alpha = 0.55;
    double max_joint_speed_mps = 8.0;
    double hold_last_valid_seconds = 0.15;
    bool verbose_pose_stats = true;
};

struct TriangulationResult {
    PointCloud cloud;           // triangulated 3D points
    Trajectory cameras;         // two camera poses (cam1 at origin, cam2 at [R|t])
    LineSet projection_lines;   // dashed lines from camera origins to 3D points
};

struct StereoCalibration {
    cv::Mat K1;                 // 3x3
    cv::Mat D1;                 // Nx1 or 1xN
    cv::Mat K2;                 // 3x3
    cv::Mat D2;                 // Nx1 or 1xN
    cv::Mat R;                  // 3x3, cam2-from-cam1 rotation
    cv::Mat T;                  // 3x1, cam2-from-cam1 translation
    cv::Size image_size;        // calibration image size
};

class TriangulationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

TriangulationResult triangulateFromImages(const std::string& image1_path,
                                          const std::string& image2_path,
                                          const TriangulationConfig& cfg = {});

StereoCalibration loadStereoCalibration(const std::string& filepath);
Pose2DSequence loadPose2DSequenceCSV(const std::string& filepath);

Pose3D triangulatePoseFromStereoKeypoints(const Pose2D& left_pose,
                                          const Pose2D& right_pose,
                                          const StereoCalibration& calib,
                                          const TriangulationConfig& cfg = {});

Pose3DSequence triangulatePoseSequenceFromStereoKeypoints(const Pose2DSequence& left_seq,
                                                          const Pose2DSequence& right_seq,
                                                          const StereoCalibration& calib,
                                                          const TriangulationConfig& cfg = {});

} // namespace vio
