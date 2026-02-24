#pragma once

#include "vio/types.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace vio {

struct PoseDetectionConfig {
    float person_confidence = 0.35f;
    float keypoint_confidence = 0.35f;
    float nms_iou_threshold = 0.45f;
    int input_width = 640;
    int input_height = 640;
};

Pose2D detectPoseYOLO(const cv::Mat& image_bgr,
                      const std::string& model_path,
                      const PoseDetectionConfig& cfg = {});

void drawPose2DOverlay(cv::Mat& image_bgr,
                       const Pose2D& pose,
                       float min_joint_confidence = 0.25f);

const std::vector<std::pair<int, int>>& coco17SkeletonEdges();

} // namespace vio
