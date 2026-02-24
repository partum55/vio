#include "vio/stereo_triangulation.h"

#include <Eigen/Core>

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

vio::StereoCalibration makeCalibration(double baseline) {
    vio::StereoCalibration calib;
    calib.K1 = (cv::Mat_<double>(3, 3) << 800.0, 0.0, 320.0,
                                           0.0, 800.0, 240.0,
                                           0.0, 0.0, 1.0);
    calib.K2 = calib.K1.clone();
    calib.D1 = cv::Mat::zeros(1, 5, CV_64F);
    calib.D2 = cv::Mat::zeros(1, 5, CV_64F);
    calib.R = cv::Mat::eye(3, 3, CV_64F);
    calib.T = (cv::Mat_<double>(3, 1) << baseline, 0.0, 0.0);
    calib.image_size = cv::Size(640, 480);
    return calib;
}

cv::Point2f project(const Eigen::Vector3d& p, const cv::Mat& K) {
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);
    return cv::Point2f(static_cast<float>(fx * (p.x() / p.z()) + cx),
                       static_cast<float>(fy * (p.y() / p.z()) + cy));
}

bool testSequenceAccuracy() {
    const vio::StereoCalibration calib = makeCalibration(0.25);
    const int kNumFrames = 60;
    const int kNumJoints = 17;

    std::vector<Eigen::Vector3d> base_joints;
    base_joints.reserve(kNumJoints);
    for (int i = 0; i < kNumJoints; ++i) {
        const double x = -0.5 + 0.06 * i;
        const double y = 1.0 + 0.03 * (i % 5);
        const double z = 4.0 + 0.04 * (i % 7);
        base_joints.emplace_back(x, y, z);
    }

    vio::Pose2DSequence left_seq;
    vio::Pose2DSequence right_seq;
    left_seq.reserve(kNumFrames);
    right_seq.reserve(kNumFrames);

    std::vector<std::unordered_map<int, Eigen::Vector3d>> gt_seq;
    gt_seq.reserve(kNumFrames);

    std::mt19937 rng(7);
    std::normal_distribution<float> pixel_noise(0.0f, 0.35f);

    for (int frame = 0; frame < kNumFrames; ++frame) {
        const double ts = frame * 0.0333333;
        vio::Pose2D left_pose;
        vio::Pose2D right_pose;
        left_pose.timestamp = ts;
        right_pose.timestamp = ts;

        std::unordered_map<int, Eigen::Vector3d> gt_frame;

        for (int id = 0; id < kNumJoints; ++id) {
            Eigen::Vector3d p = base_joints[id];
            p.x() += 0.15 * std::sin(ts * 2.5 + id * 0.2);
            p.y() += 0.08 * std::cos(ts * 1.7 + id * 0.1);
            p.z() += 0.12 * std::sin(ts * 1.2 + id * 0.15);

            const cv::Point2f uv1 = project(p, calib.K1);
            const Eigen::Vector3d p2 = p + Eigen::Vector3d(0.25, 0.0, 0.0);
            const cv::Point2f uv2 = project(p2, calib.K2);

            float conf_l = 0.95f;
            float conf_r = 0.95f;
            if (((frame + id) % 23) == 0) conf_l = 0.2f;
            if (((frame + 2 * id) % 29) == 0) conf_r = 0.2f;

            left_pose.joints.push_back({id, cv::Point2f(uv1.x + pixel_noise(rng), uv1.y + pixel_noise(rng)), conf_l});
            right_pose.joints.push_back({id, cv::Point2f(uv2.x + pixel_noise(rng), uv2.y + pixel_noise(rng)), conf_r});
            gt_frame[id] = p;
        }

        left_seq.push_back(left_pose);
        right_seq.push_back(right_pose);
        gt_seq.push_back(std::move(gt_frame));
    }

    vio::TriangulationConfig cfg;
    cfg.min_joint_confidence = 0.3;
    cfg.max_epipolar_error_px = 2.0;
    cfg.min_disparity_px = 1.5;
    cfg.max_reprojection_error_px = 2.0;
    cfg.temporal_smoothing_alpha = 0.55;
    cfg.hold_last_valid_seconds = 0.1;
    cfg.verbose_pose_stats = false;

    vio::Pose3DSequence out = vio::triangulatePoseSequenceFromStereoKeypoints(left_seq, right_seq, calib, cfg);

    int valid_count = 0;
    std::vector<double> errors;
    for (size_t i = 0; i < out.size(); ++i) {
        for (const auto& joint : out[i].joints) {
            if (!joint.valid)
                continue;
            auto it = gt_seq[i].find(joint.id);
            if (it == gt_seq[i].end())
                continue;
            ++valid_count;
            errors.push_back((joint.xyz - it->second).norm());
        }
    }

    if (errors.empty())
        return false;

    const double mpjpe = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    const double valid_ratio = static_cast<double>(valid_count) / (kNumFrames * kNumJoints);

    std::cout << "  sequence accuracy: MPJPE=" << mpjpe
              << "m, valid ratio=" << (100.0 * valid_ratio) << "%\n";

    return mpjpe < 0.12 && valid_ratio > 0.75;
}

bool testLowDisparityRejection() {
    vio::StereoCalibration calib = makeCalibration(0.005);

    vio::Pose2D left_pose;
    vio::Pose2D right_pose;
    left_pose.timestamp = 0.0;
    right_pose.timestamp = 0.0;

    for (int id = 0; id < 10; ++id) {
        const Eigen::Vector3d p(-0.3 + 0.06 * id, 1.2, 6.0);
        const cv::Point2f uv1 = project(p, calib.K1);
        const Eigen::Vector3d p2 = p + Eigen::Vector3d(0.005, 0.0, 0.0);
        const cv::Point2f uv2 = project(p2, calib.K2);
        left_pose.joints.push_back({id, uv1, 0.99f});
        right_pose.joints.push_back({id, uv2, 0.99f});
    }

    vio::TriangulationConfig cfg;
    cfg.min_joint_confidence = 0.2;
    cfg.min_disparity_px = 2.0;
    cfg.max_epipolar_error_px = 2.0;
    cfg.max_reprojection_error_px = 2.0;

    auto out = vio::triangulatePoseFromStereoKeypoints(left_pose, right_pose, calib, cfg);
    int valid = 0;
    for (const auto& j : out.joints) {
        if (j.valid) ++valid;
    }
    std::cout << "  low-disparity rejection: valid joints=" << valid << "\n";
    return valid == 0;
}

bool testLoaders() {
    const std::string calib_path = "/tmp/vio_stereo_calib.yml";
    const std::string kp_path = "/tmp/vio_kp.csv";

    {
        std::ofstream ofs(calib_path);
        ofs << "%YAML:1.0\n";
        ofs << "image_width: 640\n";
        ofs << "image_height: 480\n";
        ofs << "K1: !!opencv-matrix\n";
        ofs << "   rows: 3\n";
        ofs << "   cols: 3\n";
        ofs << "   dt: d\n";
        ofs << "   data: [800, 0, 320, 0, 800, 240, 0, 0, 1]\n";
        ofs << "K2: !!opencv-matrix\n";
        ofs << "   rows: 3\n";
        ofs << "   cols: 3\n";
        ofs << "   dt: d\n";
        ofs << "   data: [800, 0, 320, 0, 800, 240, 0, 0, 1]\n";
        ofs << "D1: !!opencv-matrix\n";
        ofs << "   rows: 1\n";
        ofs << "   cols: 5\n";
        ofs << "   dt: d\n";
        ofs << "   data: [0, 0, 0, 0, 0]\n";
        ofs << "D2: !!opencv-matrix\n";
        ofs << "   rows: 1\n";
        ofs << "   cols: 5\n";
        ofs << "   dt: d\n";
        ofs << "   data: [0, 0, 0, 0, 0]\n";
        ofs << "R: !!opencv-matrix\n";
        ofs << "   rows: 3\n";
        ofs << "   cols: 3\n";
        ofs << "   dt: d\n";
        ofs << "   data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n";
        ofs << "T: !!opencv-matrix\n";
        ofs << "   rows: 3\n";
        ofs << "   cols: 1\n";
        ofs << "   dt: d\n";
        ofs << "   data: [0.2, 0, 0]\n";
    }

    {
        std::ofstream ofs(kp_path);
        ofs << "# ts,id,x,y,conf\n";
        ofs << "0.0,0,100,120,0.9\n";
        ofs << "0.0,1,110,130,0.8\n";
        ofs << "0.033,0,101,121,0.95\n";
        ofs << "0.033,1,111,131,0.85\n";
    }

    auto calib = vio::loadStereoCalibration(calib_path);
    auto seq = vio::loadPose2DSequenceCSV(kp_path);

    const bool calib_ok = !calib.K1.empty() && calib.image_size.width == 640 && calib.image_size.height == 480;
    const bool seq_ok = seq.size() == 2 && seq[0].joints.size() == 2 && seq[1].joints.size() == 2;
    return calib_ok && seq_ok;
}

} // namespace

int main() {
    int passed = 0;
    int total = 0;

    std::cout << "=== Pose Triangulation Tests ===\n\n";

    {
        ++total;
        const bool ok = testSequenceAccuracy();
        std::cout << (ok ? "[PASS] " : "[FAIL] ")
                  << "Sequence accuracy + smoothing\n\n";
        if (ok) ++passed;
    }

    {
        ++total;
        const bool ok = testLowDisparityRejection();
        std::cout << (ok ? "[PASS] " : "[FAIL] ")
                  << "Low disparity rejection\n\n";
        if (ok) ++passed;
    }

    {
        ++total;
        const bool ok = testLoaders();
        std::cout << (ok ? "[PASS] " : "[FAIL] ")
                  << "Calibration + CSV loaders\n\n";
        if (ok) ++passed;
    }

    std::cout << "=== Summary: " << passed << " / " << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
