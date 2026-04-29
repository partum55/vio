#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace vio {

struct FrameState {
    int frame_id = -1;
    double timestamp = 0.0;
    // Camera pose convention used by geometry/pipeline:
    // q_wc and t_wc transform a point from camera frame C into world frame W:
    // p_w = R_wc * p_c + t_wc. PnP stores global camera pose in this form.
    Eigen::Quaterniond q_wc = Eigen::Quaterniond(1, 0, 0, 0);
    Eigen::Vector3d t_wc = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_w = Eigen::Vector3d::Zero();
};

struct Observation {
    int frame_id = -1;
    int track_id = -1;
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    bool valid = true;
};

struct TrackedFrame {
    FrameState state;
    std::vector<Observation> observations;
};

struct CameraIntrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    int width = 0;
    int height = 0;

    Eigen::Matrix3d matrix() const {
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(0, 2) = cx;
        K(1, 2) = cy;
        return K;
    }

    bool isValid() const {
        return fx > 0.0 && fy > 0.0;
    }
};

struct Landmark {
    int id = -1;
    int track_id = -1;
    Eigen::Vector3d p_w = Eigen::Vector3d::Zero();
    bool valid = false;
    double reprojection_error = 0.0;
    int num_observations = 0;
};

struct Track {
    int id = -1;
    cv::Point2f pt;
    std::vector<cv::Point2f> history;
};

} // namespace vio
