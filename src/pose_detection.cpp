#include "vio/pose_detection.h"

#include "vio/stereo_triangulation.h"

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <utility>
#include <vector>

namespace vio {

namespace {

struct LetterboxMeta {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    int out_w = 0;
    int out_h = 0;
};

cv::Mat letterbox(const cv::Mat& image, int out_w, int out_h, LetterboxMeta& meta) {
    if (image.empty())
        throw TriangulationError("Pose detection input image is empty");

    const float scale = std::min(static_cast<float>(out_w) / static_cast<float>(image.cols),
                                 static_cast<float>(out_h) / static_cast<float>(image.rows));
    const int new_w = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(image.rows * scale)));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(out_h, out_w, image.type(), cv::Scalar(114, 114, 114));
    const int pad_x = (out_w - new_w) / 2;
    const int pad_y = (out_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    meta.scale = scale;
    meta.pad_x = static_cast<float>(pad_x);
    meta.pad_y = static_cast<float>(pad_y);
    meta.out_w = out_w;
    meta.out_h = out_h;
    return canvas;
}

cv::Mat normalizeDetections(const cv::Mat& output) {
    // Expected YOLOv8 pose output is either [1, N, 56] or [1, 56, N].
    if (output.dims == 3) {
        const int d1 = output.size[1];
        const int d2 = output.size[2];
        if (d2 == 56) {
            return output.reshape(1, d1).clone();
        }
        if (d1 == 56) {
            cv::Mat ch_first(56, d2, CV_32F, const_cast<float*>(output.ptr<float>()));
            cv::Mat transposed;
            cv::transpose(ch_first, transposed);
            return transposed.clone();
        }
    }

    if (output.dims == 2 && output.cols == 56)
        return output.clone();

    throw TriangulationError("Unsupported pose model output shape. Expected Nx56 detections.");
}

cv::Point2f unletterboxPoint(float x, float y, const LetterboxMeta& meta, int img_w, int img_h) {
    float px = (x - meta.pad_x) / meta.scale;
    float py = (y - meta.pad_y) / meta.scale;
    px = std::clamp(px, 0.0f, static_cast<float>(img_w - 1));
    py = std::clamp(py, 0.0f, static_cast<float>(img_h - 1));
    return cv::Point2f(px, py);
}

} // namespace

Pose2D detectPoseYOLO(const cv::Mat& image_bgr,
                      const std::string& model_path,
                      const PoseDetectionConfig& cfg) {
    cv::dnn::Net net = cv::dnn::readNet(model_path);
    if (net.empty())
        throw TriangulationError("Failed to load pose model: " + model_path);

    LetterboxMeta meta;
    cv::Mat padded = letterbox(image_bgr, cfg.input_width, cfg.input_height, meta);

    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0,
                                          cv::Size(cfg.input_width, cfg.input_height),
                                          cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);
    cv::Mat output = net.forward();
    cv::Mat det = normalizeDetections(output);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::vector<Joint2D>> candidates;

    boxes.reserve(det.rows);
    scores.reserve(det.rows);
    candidates.reserve(det.rows);

    for (int i = 0; i < det.rows; ++i) {
        const float* row = det.ptr<float>(i);
        const float conf = row[4];
        if (conf < cfg.person_confidence)
            continue;

        const float cx = row[0];
        const float cy = row[1];
        const float w = row[2];
        const float h = row[3];
        const float x0 = cx - 0.5f * w;
        const float y0 = cy - 0.5f * h;
        boxes.emplace_back(static_cast<int>(std::round(x0)),
                           static_cast<int>(std::round(y0)),
                           std::max(1, static_cast<int>(std::round(w))),
                           std::max(1, static_cast<int>(std::round(h))));
        scores.push_back(conf);

        std::vector<Joint2D> joints;
        joints.reserve(17);
        for (int j = 0; j < 17; ++j) {
            const float kx = row[5 + 3 * j];
            const float ky = row[5 + 3 * j + 1];
            const float kc = row[5 + 3 * j + 2];
            const cv::Point2f uv = unletterboxPoint(kx, ky, meta, image_bgr.cols, image_bgr.rows);
            joints.push_back({j, uv, kc});
        }
        candidates.push_back(std::move(joints));
    }

    if (boxes.empty())
        throw TriangulationError("No person pose detected in image using model: " + model_path);

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, cfg.person_confidence, cfg.nms_iou_threshold, keep);
    if (keep.empty())
        throw TriangulationError("No pose detections survived NMS.");

    int best_idx = keep[0];
    for (int idx : keep) {
        if (scores[idx] > scores[best_idx])
            best_idx = idx;
    }

    Pose2D pose;
    pose.timestamp = 0.0;
    pose.joints.reserve(candidates[best_idx].size());
    for (const auto& joint : candidates[best_idx]) {
        if (joint.confidence >= cfg.keypoint_confidence)
            pose.joints.push_back(joint);
    }

    if (pose.joints.empty())
        throw TriangulationError("Detected person has no keypoints above confidence threshold.");

    return pose;
}

const std::vector<std::pair<int, int>>& coco17SkeletonEdges() {
    static const std::vector<std::pair<int, int>> edges = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {5, 11}, {6, 12}, {11, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16},
        {0, 5}, {0, 6}
    };
    return edges;
}

void drawPose2DOverlay(cv::Mat& image_bgr,
                       const Pose2D& pose,
                       float min_joint_confidence) {
    std::vector<const Joint2D*> by_id(17, nullptr);
    for (const auto& joint : pose.joints) {
        if (joint.id < 0 || joint.id >= static_cast<int>(by_id.size()))
            continue;
        if (joint.confidence < min_joint_confidence)
            continue;
        by_id[joint.id] = &joint;
    }

    for (const auto& edge : coco17SkeletonEdges()) {
        const Joint2D* a = by_id[edge.first];
        const Joint2D* b = by_id[edge.second];
        if (!a || !b)
            continue;
        cv::line(image_bgr, a->uv, b->uv, cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
    }

    for (const auto* joint : by_id) {
        if (!joint)
            continue;
        cv::circle(image_bgr, joint->uv, 4, cv::Scalar(255, 120, 0), -1, cv::LINE_AA);
    }
}

} // namespace vio
