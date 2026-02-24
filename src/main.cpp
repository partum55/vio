#include "vio/data_generator.h"
#include "vio/data_loader.h"
#include "vio/pose_detection.h"
#include "vio/stereo_triangulation.h"
#include "vio/viewer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

const std::array<Eigen::Vector3f, 12> kPalette = {
    Eigen::Vector3f(0.95f, 0.26f, 0.21f),
    Eigen::Vector3f(0.25f, 0.32f, 0.71f),
    Eigen::Vector3f(0.20f, 0.66f, 0.33f),
    Eigen::Vector3f(1.00f, 0.60f, 0.00f),
    Eigen::Vector3f(0.55f, 0.34f, 0.29f),
    Eigen::Vector3f(0.61f, 0.15f, 0.69f),
    Eigen::Vector3f(0.00f, 0.67f, 0.76f),
    Eigen::Vector3f(0.76f, 0.18f, 0.20f),
    Eigen::Vector3f(0.30f, 0.69f, 0.31f),
    Eigen::Vector3f(1.00f, 0.76f, 0.03f),
    Eigen::Vector3f(0.13f, 0.59f, 0.95f),
    Eigen::Vector3f(0.91f, 0.12f, 0.39f),
};

Eigen::Vector3f colorForJoint(int joint_id) {
    const size_t idx = static_cast<size_t>(std::abs(joint_id)) % kPalette.size();
    return kPalette[idx];
}

double median(std::vector<double> values) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0)
        return 0.5 * (values[mid - 1] + values[mid]);
    return values[mid];
}

void appendPose3DToScene(const vio::Pose3D& pose, vio::PointCloud& cloud, vio::LineSet& lines) {
    std::unordered_map<int, const vio::Joint3D*> by_id;
    by_id.reserve(pose.joints.size());
    for (const auto& joint : pose.joints) {
        if (!joint.valid)
            continue;
        by_id[joint.id] = &joint;
        cloud.push_back({joint.xyz, colorForJoint(joint.id)});
    }

    for (const auto& edge : vio::coco17SkeletonEdges()) {
        auto a_it = by_id.find(edge.first);
        auto b_it = by_id.find(edge.second);
        if (a_it == by_id.end() || b_it == by_id.end())
            continue;

        const auto* a = a_it->second;
        const auto* b = b_it->second;
        lines.push_back({a->xyz, b->xyz, Eigen::Vector3f(0.15f, 0.95f, 0.95f)});
    }
}

vio::Trajectory makeStereoCameraTrajectory(const vio::StereoCalibration& calib) {
    cv::Mat R;
    cv::Mat T;
    calib.R.convertTo(R, CV_64F);
    calib.T.convertTo(T, CV_64F);
    if (T.rows == 1 && T.cols == 3)
        T = T.t();

    vio::Trajectory cams;
    cams.reserve(2);

    vio::CameraPose c1;
    c1.timestamp = 0.0;
    c1.T_wc = Eigen::Matrix4d::Identity();
    cams.push_back(c1);

    vio::CameraPose c2;
    c2.timestamp = 1.0;
    c2.T_wc = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            c2.T_wc(r, c) = R.at<double>(r, c);
        c2.T_wc(r, 3) = T.at<double>(r, 0);
    }
    cams.push_back(c2);

    return cams;
}

void adaptCalibrationToImageSize(vio::StereoCalibration& calib, const cv::Size& image_size) {
    if (image_size.width <= 0 || image_size.height <= 0)
        return;

    if (calib.image_size.width <= 0 || calib.image_size.height <= 0) {
        calib.image_size = image_size;
        return;
    }

    if (calib.image_size == image_size)
        return;

    const double sx = static_cast<double>(image_size.width) / calib.image_size.width;
    const double sy = static_cast<double>(image_size.height) / calib.image_size.height;

    auto scaleK = [sx, sy](cv::Mat& K) {
        cv::Mat K64;
        K.convertTo(K64, CV_64F);
        K64.at<double>(0, 0) *= sx; // fx
        K64.at<double>(0, 2) *= sx; // cx
        K64.at<double>(1, 1) *= sy; // fy
        K64.at<double>(1, 2) *= sy; // cy
        K = K64;
    };

    scaleK(calib.K1);
    scaleK(calib.K2);
    std::cout << "Calibration image size mismatch detected ("
              << calib.image_size.width << "x" << calib.image_size.height << " -> "
              << image_size.width << "x" << image_size.height
              << "). Rescaled intrinsics for this run.\n";
    calib.image_size = image_size;
}

} // namespace

static void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "\n"
        "Options:\n"
        "  --help, -h              Show this help message\n"
        "  --trajectory <file>     Load trajectory from TUM format file\n"
        "  --cloud <file>          Load point cloud from XYZ text file\n"
        "  --record <file>         Record to video file (default: output.mp4)\n"
        "  --width <pixels>        Video width  (default: 1920)\n"
        "  --height <pixels>       Video height (default: 1080)\n"
        "  --fps <rate>            Video FPS    (default: 60)\n"
        "  --image1 <file>         First image for stereo triangulation\n"
        "  --image2 <file>         Second image for stereo triangulation\n"
        "  --stereo-calib <file>   Stereo calibration file (YAML/XML)\n"
        "  --left-kp <file>        Left-camera keypoints CSV: timestamp,id,x,y,confidence\n"
        "  --right-kp <file>       Right-camera keypoints CSV: timestamp,id,x,y,confidence\n"
        "  --pose-model <file>     YOLO pose ONNX model for 2D keypoint extraction\n"
        "  --show-pose-2d          Show 2D pose overlay before launching 3D view\n"
        "\n"
        "If no --trajectory, --cloud, --image1/--image2, keypoint/calib, or pose-model is given,\n"
        "a synthetic demo is shown.\n"
        "\n"
        "File formats:\n"
        "  Trajectory (TUM):  timestamp tx ty tz qx qy qz qw\n"
        "  Point cloud (XYZ): x y z [r g b]   (RGB 0-255, optional)\n"
        "  Keypoints CSV:     timestamp,id,x,y[,confidence]\n"
        "  Lines starting with '#' are treated as comments.\n"
        "\n"
        "Examples:\n"
        "  " << prog << "                                       # synthetic demo\n"
        "  " << prog << " --trajectory traj.tum                 # trajectory only\n"
        "  " << prog << " --cloud points.xyz                    # point cloud only\n"
        "  " << prog << " --trajectory t.tum --cloud c.xyz      # both\n"
        "  " << prog << " --record out.mp4 --trajectory t.tum   # record with data\n"
        "  " << prog << " --image1 left.jpg --image2 right.jpg   # sparse stereo triangulation\n"
        "  " << prog << " --stereo-calib stereo.yml --left-kp left.csv --right-kp right.csv\n"
        "  " << prog << " --image1 left.jpg --image2 right.jpg --stereo-calib stereo.yml \\\n"
        "      --pose-model yolov8n-pose.onnx --show-pose-2d\n";
}

int main(int argc, char** argv) {
    std::string trajectory_path;
    std::string cloud_path;
    std::string record_path;
    std::string image1_path;
    std::string image2_path;
    std::string stereo_calib_path;
    std::string left_kp_path;
    std::string right_kp_path;
    std::string pose_model_path;
    bool show_pose_2d = false;
    int width = 1920;
    int height = 1080;
    int fps = 60;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--trajectory") {
            if (++i >= argc) { std::cerr << "Error: --trajectory requires a file path\n"; return 1; }
            trajectory_path = argv[i];
        } else if (arg == "--cloud") {
            if (++i >= argc) { std::cerr << "Error: --cloud requires a file path\n"; return 1; }
            cloud_path = argv[i];
        } else if (arg == "--record") {
            if (++i >= argc) { std::cerr << "Error: --record requires a file path\n"; return 1; }
            record_path = argv[i];
        } else if (arg == "--width") {
            if (++i >= argc) { std::cerr << "Error: --width requires a value\n"; return 1; }
            width = std::stoi(argv[i]);
        } else if (arg == "--height") {
            if (++i >= argc) { std::cerr << "Error: --height requires a value\n"; return 1; }
            height = std::stoi(argv[i]);
        } else if (arg == "--fps") {
            if (++i >= argc) { std::cerr << "Error: --fps requires a value\n"; return 1; }
            fps = std::stoi(argv[i]);
        } else if (arg == "--image1") {
            if (++i >= argc) { std::cerr << "Error: --image1 requires a file path\n"; return 1; }
            image1_path = argv[i];
        } else if (arg == "--image2") {
            if (++i >= argc) { std::cerr << "Error: --image2 requires a file path\n"; return 1; }
            image2_path = argv[i];
        } else if (arg == "--stereo-calib") {
            if (++i >= argc) { std::cerr << "Error: --stereo-calib requires a file path\n"; return 1; }
            stereo_calib_path = argv[i];
        } else if (arg == "--left-kp") {
            if (++i >= argc) { std::cerr << "Error: --left-kp requires a file path\n"; return 1; }
            left_kp_path = argv[i];
        } else if (arg == "--right-kp") {
            if (++i >= argc) { std::cerr << "Error: --right-kp requires a file path\n"; return 1; }
            right_kp_path = argv[i];
        } else if (arg == "--pose-model") {
            if (++i >= argc) { std::cerr << "Error: --pose-model requires a file path\n"; return 1; }
            pose_model_path = argv[i];
        } else if (arg == "--show-pose-2d") {
            show_pose_2d = true;
        } else {
            std::cerr << "Error: unknown option '" << arg << "'\n\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate stereo image args
    if (image1_path.empty() != image2_path.empty()) {
        std::cerr << "Error: --image1 and --image2 must both be provided\n";
        return 1;
    }

    const bool use_pose_csv_mode = !left_kp_path.empty() || !right_kp_path.empty();
    if (use_pose_csv_mode &&
        (stereo_calib_path.empty() || left_kp_path.empty() || right_kp_path.empty())) {
        std::cerr << "Error: --stereo-calib, --left-kp, and --right-kp must all be provided together\n";
        return 1;
    }

    const bool use_pose_image_mode = !pose_model_path.empty();
    if (show_pose_2d && !use_pose_image_mode) {
        std::cerr << "Error: --show-pose-2d requires --pose-model mode\n";
        return 1;
    }
    if (use_pose_image_mode &&
        (stereo_calib_path.empty() || image1_path.empty() || image2_path.empty())) {
        std::cerr << "Error: --pose-model requires --image1, --image2, and --stereo-calib\n";
        return 1;
    }
    if (use_pose_image_mode && use_pose_csv_mode) {
        std::cerr << "Error: choose one pose input mode: either --pose-model or --left-kp/--right-kp\n";
        return 1;
    }

    // Load or generate data
    bool use_synthetic = trajectory_path.empty() && cloud_path.empty() &&
                         image1_path.empty() && !use_pose_csv_mode && !use_pose_image_mode;

    vio::Trajectory trajectory;
    vio::PointCloud cloud;
    vio::LineSet lines;

    if (use_synthetic) {
        std::cout << "No data files specified — running synthetic demo.\n";
        vio::GeneratorConfig gen_cfg;
        trajectory = vio::generateTrajectory(gen_cfg);
        cloud = vio::generatePointCloud(gen_cfg);
    } else {
        if (!trajectory_path.empty())
            trajectory = vio::loadTrajectoryTUM(trajectory_path);
        if (!cloud_path.empty())
            cloud = vio::loadPointCloudXYZ(cloud_path);

        if (use_pose_image_mode) {
            try {
                auto calib = vio::loadStereoCalibration(stereo_calib_path);
                cv::Mat left_img = cv::imread(image1_path, cv::IMREAD_COLOR);
                cv::Mat right_img = cv::imread(image2_path, cv::IMREAD_COLOR);
                if (left_img.empty())
                    throw vio::TriangulationError("Failed to load image: " + image1_path);
                if (right_img.empty())
                    throw vio::TriangulationError("Failed to load image: " + image2_path);
                if (left_img.size() != right_img.size())
                    throw vio::TriangulationError("Left/right image sizes differ. Rectified stereo requires same size.");

                adaptCalibrationToImageSize(calib, left_img.size());

                vio::PoseDetectionConfig det_cfg;
                det_cfg.person_confidence = 0.35f;
                det_cfg.keypoint_confidence = 0.30f;
                det_cfg.nms_iou_threshold = 0.45f;
                det_cfg.input_width = 640;
                det_cfg.input_height = 640;

                vio::Pose2D left_pose = vio::detectPoseYOLO(left_img, pose_model_path, det_cfg);
                vio::Pose2D right_pose = vio::detectPoseYOLO(right_img, pose_model_path, det_cfg);

                if (show_pose_2d) {
                    cv::Mat left_vis = left_img.clone();
                    cv::Mat right_vis = right_img.clone();
                    vio::drawPose2DOverlay(left_vis, left_pose, 0.25f);
                    vio::drawPose2DOverlay(right_vis, right_pose, 0.25f);
                    cv::Mat combined;
                    cv::hconcat(left_vis, right_vis, combined);
                    cv::imshow("2D Pose Detections (Left | Right)", combined);
                    std::cout << "Press any key on the 2D pose window to launch 3D visualization...\n";
                    cv::waitKey(0);
                    cv::destroyAllWindows();
                }

                vio::TriangulationConfig pose_cfg;
                pose_cfg.show_matches = false;
                pose_cfg.verbose_pose_stats = false;
                pose_cfg.min_joint_confidence = 0.30;
                pose_cfg.max_epipolar_error_px = 2.0;
                pose_cfg.min_disparity_px = 1.5;
                pose_cfg.max_reprojection_error_px = 2.5;

                const vio::Pose3D pose3d = vio::triangulatePoseFromStereoKeypoints(
                    left_pose, right_pose, calib, pose_cfg);

                appendPose3DToScene(pose3d, cloud, lines);
                const auto cams = makeStereoCameraTrajectory(calib);
                trajectory.insert(trajectory.end(), cams.begin(), cams.end());

                std::vector<double> reproj_errors;
                int valid_joint_count = 0;
                for (const auto& joint : pose3d.joints) {
                    if (!joint.valid)
                        continue;
                    ++valid_joint_count;
                    if (std::isfinite(joint.reproj_err_px))
                        reproj_errors.push_back(joint.reproj_err_px);
                }

                std::cout << "Pose from 2 images complete:\n"
                          << "  valid 3D joints: " << valid_joint_count << "\n"
                          << "  median reprojection error: " << median(reproj_errors) << " px\n";
            } catch (const vio::TriangulationError& e) {
                std::cerr << "Pose-from-images error: " << e.what() << "\n";
                return 1;
            }
        } else if (!image1_path.empty()) {
            try {
                auto result = vio::triangulateFromImages(image1_path, image2_path);
                cloud.insert(cloud.end(), result.cloud.begin(), result.cloud.end());
                trajectory.insert(trajectory.end(), result.cameras.begin(), result.cameras.end());
                lines.insert(lines.end(), result.projection_lines.begin(), result.projection_lines.end());
            } catch (const vio::TriangulationError& e) {
                std::cerr << "Stereo triangulation error: " << e.what() << "\n";
                return 1;
            }
        }

        if (use_pose_csv_mode) {
            try {
                auto calib = vio::loadStereoCalibration(stereo_calib_path);
                auto left_seq = vio::loadPose2DSequenceCSV(left_kp_path);
                auto right_seq = vio::loadPose2DSequenceCSV(right_kp_path);

                vio::TriangulationConfig pose_cfg;
                pose_cfg.show_matches = false;
                auto pose_seq = vio::triangulatePoseSequenceFromStereoKeypoints(left_seq, right_seq, calib, pose_cfg);

                int valid_joint_count = 0;
                for (const auto& pose : pose_seq) {
                    Eigen::Vector3d center = Eigen::Vector3d::Zero();
                    int center_count = 0;
                    for (const auto& joint : pose.joints) {
                        if (!joint.valid)
                            continue;

                        ++valid_joint_count;
                        cloud.push_back({joint.xyz, colorForJoint(joint.id)});
                        center += joint.xyz;
                        ++center_count;
                    }

                    if (center_count > 0) {
                        vio::CameraPose cam;
                        cam.timestamp = pose.timestamp;
                        cam.T_wc = Eigen::Matrix4d::Identity();
                        cam.T_wc.block<3, 1>(0, 3) = center / center_count;
                        trajectory.push_back(cam);
                    }
                }

                std::cout << "Pose triangulation complete:\n"
                          << "  frames: " << pose_seq.size() << "\n"
                          << "  valid joints: " << valid_joint_count << "\n";
            } catch (const vio::TriangulationError& e) {
                std::cerr << "Pose triangulation error: " << e.what() << "\n";
                return 1;
            }
        }
    }

    vio::Viewer viewer(trajectory, cloud, lines);

    if (!record_path.empty()) {
        vio::RecordConfig rec_cfg;
        rec_cfg.output_path = record_path;
        rec_cfg.width = width;
        rec_cfg.height = height;
        rec_cfg.fps = fps;
        std::cout << "Recording video to: " << rec_cfg.output_path << "\n";
        viewer.record(rec_cfg);
    } else {
        viewer.run();
    }

    return 0;
}
