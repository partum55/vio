#include "vio/stereo_triangulation.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vio {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-9;

struct StereoRuntime {
    cv::Mat K1;
    cv::Mat D1;
    cv::Mat K2;
    cv::Mat D2;
    cv::Mat R;
    cv::Mat T;

    cv::Mat R1_rect;
    cv::Mat R2_rect;
    cv::Mat P1_rect;
    cv::Mat P2_rect;
    cv::Mat Q;

    cv::Mat P1_norm;
    cv::Mat P2_norm;
    cv::Mat F;

    Eigen::Matrix3d R_e = Eigen::Matrix3d::Identity();
    Eigen::Vector3d T_e = Eigen::Vector3d::Zero();
};

struct CandidateMatch {
    int id = -1;
    float left_conf = 0.0f;
    float right_conf = 0.0f;
    int out_index = -1;
    cv::Point2f left_px;
    cv::Point2f right_px;
};

struct TrackState {
    bool has_value = false;
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
    double timestamp = 0.0;
    float confidence = 0.0f;
};

cv::Mat ensureMat64(const cv::Mat& src, const char* name, int rows = -1, int cols = -1) {
    if (src.empty())
        throw TriangulationError(std::string("Missing calibration matrix: ") + name);

    cv::Mat out;
    src.convertTo(out, CV_64F);

    if (rows > 0 && out.rows != rows)
        throw TriangulationError(std::string("Calibration matrix ") + name + " has invalid row count");
    if (cols > 0 && out.cols != cols)
        throw TriangulationError(std::string("Calibration matrix ") + name + " has invalid column count");

    return out;
}

cv::Mat ensureDistortion64(const cv::Mat& src, const char* name) {
    if (src.empty())
        return cv::Mat::zeros(1, 5, CV_64F);

    cv::Mat out;
    src.convertTo(out, CV_64F);
    if (out.rows != 1 && out.cols != 1)
        out = out.reshape(1, 1);
    if (out.rows != 1)
        out = out.reshape(1, 1);
    return out;
}

cv::Mat makeSkew(const cv::Mat& t_3x1) {
    return (cv::Mat_<double>(3, 3) << 0.0, -t_3x1.at<double>(2, 0), t_3x1.at<double>(1, 0),
                                       t_3x1.at<double>(2, 0), 0.0, -t_3x1.at<double>(0, 0),
                                      -t_3x1.at<double>(1, 0), t_3x1.at<double>(0, 0), 0.0);
}

cv::Mat readFirstMatrix(const cv::FileStorage& fs, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        cv::FileNode node = fs[key];
        if (!node.empty()) {
            cv::Mat m;
            node >> m;
            if (!m.empty())
                return m;
        }
    }
    return cv::Mat();
}

StereoRuntime buildRuntime(const StereoCalibration& calib) {
    StereoRuntime rt;

    rt.K1 = ensureMat64(calib.K1, "K1", 3, 3);
    rt.K2 = ensureMat64(calib.K2, "K2", 3, 3);
    rt.D1 = ensureDistortion64(calib.D1, "D1");
    rt.D2 = ensureDistortion64(calib.D2, "D2");
    rt.R = ensureMat64(calib.R, "R", 3, 3);

    cv::Mat T = ensureMat64(calib.T, "T");
    if (T.rows == 1 && T.cols == 3)
        T = T.t();
    if (T.rows != 3 || T.cols != 1)
        throw TriangulationError("Calibration matrix T must be 3x1 or 1x3");
    rt.T = T;

    if (calib.image_size.width <= 0 || calib.image_size.height <= 0)
        throw TriangulationError("Calibration image_size must be set and positive");

    cv::stereoRectify(rt.K1, rt.D1, rt.K2, rt.D2, calib.image_size, rt.R, rt.T,
                      rt.R1_rect, rt.R2_rect, rt.P1_rect, rt.P2_rect, rt.Q,
                      cv::CALIB_ZERO_DISPARITY, 0.0, calib.image_size);

    rt.P1_norm = cv::Mat::eye(3, 4, CV_64F);
    cv::hconcat(rt.R, rt.T, rt.P2_norm);

    cv::Mat E = makeSkew(rt.T) * rt.R;
    cv::Mat K1_inv = rt.K1.inv();
    cv::Mat K2_inv = rt.K2.inv();
    rt.F = K2_inv.t() * E * K1_inv;

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            rt.R_e(r, c) = rt.R.at<double>(r, c);
        rt.T_e(r) = rt.T.at<double>(r, 0);
    }

    return rt;
}

double huberWeight(double residual_abs, double delta) {
    if (residual_abs <= delta || residual_abs < kEps)
        return 1.0;
    return delta / residual_abs;
}

double huberCost(double residual, double delta) {
    const double a = std::abs(residual);
    if (a <= delta)
        return 0.5 * a * a;
    return delta * (a - 0.5 * delta);
}

std::unordered_map<int, Joint2D> dedupeByBestConfidence(const std::vector<Joint2D>& joints) {
    std::unordered_map<int, Joint2D> by_id;
    for (const auto& joint : joints) {
        auto it = by_id.find(joint.id);
        if (it == by_id.end() || joint.confidence > it->second.confidence)
            by_id[joint.id] = joint;
    }
    return by_id;
}

double epipolarDistancePx(const cv::Mat& F, const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Mat p1h = (cv::Mat_<double>(3, 1) << p1.x, p1.y, 1.0);
    cv::Mat p2h = (cv::Mat_<double>(3, 1) << p2.x, p2.y, 1.0);

    cv::Mat l2 = F * p1h;
    cv::Mat l1 = F.t() * p2h;

    const double num2 = std::abs(l2.at<double>(0, 0) * p2.x + l2.at<double>(1, 0) * p2.y + l2.at<double>(2, 0));
    const double den2 = std::sqrt(l2.at<double>(0, 0) * l2.at<double>(0, 0) +
                                  l2.at<double>(1, 0) * l2.at<double>(1, 0)) + kEps;

    const double num1 = std::abs(l1.at<double>(0, 0) * p1.x + l1.at<double>(1, 0) * p1.y + l1.at<double>(2, 0));
    const double den1 = std::sqrt(l1.at<double>(0, 0) * l1.at<double>(0, 0) +
                                  l1.at<double>(1, 0) * l1.at<double>(1, 0)) + kEps;

    return 0.5 * (num1 / den1 + num2 / den2);
}

bool computeReprojectionResidual(const Eigen::Vector3d& x_cam1,
                                 const StereoRuntime& rt,
                                 const cv::Point2f& obs1,
                                 const cv::Point2f& obs2,
                                 Eigen::Matrix<double, 4, 1>* residual,
                                 double* mean_err = nullptr,
                                 double* max_err = nullptr) {
    if (x_cam1.z() <= kEps)
        return false;

    Eigen::Vector3d x_cam2 = rt.R_e * x_cam1 + rt.T_e;
    if (x_cam2.z() <= kEps)
        return false;

    std::vector<cv::Point3d> obj1 = {cv::Point3d(x_cam1.x(), x_cam1.y(), x_cam1.z())};
    std::vector<cv::Point3d> obj2 = {cv::Point3d(x_cam2.x(), x_cam2.y(), x_cam2.z())};
    std::vector<cv::Point2d> proj1;
    std::vector<cv::Point2d> proj2;

    cv::projectPoints(obj1, cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0), rt.K1, rt.D1, proj1);
    cv::projectPoints(obj2, cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0), rt.K2, rt.D2, proj2);

    if (residual) {
        (*residual)(0) = proj1[0].x - obs1.x;
        (*residual)(1) = proj1[0].y - obs1.y;
        (*residual)(2) = proj2[0].x - obs2.x;
        (*residual)(3) = proj2[0].y - obs2.y;
    }

    const double err1 = std::hypot(proj1[0].x - obs1.x, proj1[0].y - obs1.y);
    const double err2 = std::hypot(proj2[0].x - obs2.x, proj2[0].y - obs2.y);
    if (mean_err)
        *mean_err = 0.5 * (err1 + err2);
    if (max_err)
        *max_err = std::max(err1, err2);
    return true;
}

double robustResidualCost(const Eigen::Matrix<double, 4, 1>& residual, double delta) {
    double cost = 0.0;
    for (int i = 0; i < 4; ++i)
        cost += huberCost(residual(i), delta);
    return cost;
}

Eigen::Vector3d refineByReprojection(const Eigen::Vector3d& initial,
                                     const StereoRuntime& rt,
                                     const cv::Point2f& obs1,
                                     const cv::Point2f& obs2,
                                     double huber_delta_px) {
    Eigen::Vector3d x = initial;
    const double eps = 1e-5;

    Eigen::Matrix<double, 4, 1> residual;
    if (!computeReprojectionResidual(x, rt, obs1, obs2, &residual))
        return initial;

    double current_cost = robustResidualCost(residual, huber_delta_px);

    for (int iter = 0; iter < 15; ++iter) {
        Eigen::Matrix<double, 4, 3> J;
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d xp = x;
            xp(axis) += eps;
            Eigen::Matrix<double, 4, 1> rp;
            if (!computeReprojectionResidual(xp, rt, obs1, obs2, &rp))
                return x;
            J.col(axis) = (rp - residual) / eps;
        }

        Eigen::Matrix4d W = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 4; ++i)
            W(i, i) = huberWeight(std::abs(residual(i)), huber_delta_px);

        Eigen::Matrix3d H = J.transpose() * W * J + 1e-6 * Eigen::Matrix3d::Identity();
        Eigen::Vector3d g = J.transpose() * W * residual;
        Eigen::Vector3d dx = -H.ldlt().solve(g);

        if (!dx.allFinite())
            break;

        bool accepted = false;
        double step = 1.0;
        for (int ls = 0; ls < 8; ++ls) {
            Eigen::Vector3d x_try = x + step * dx;
            Eigen::Matrix<double, 4, 1> r_try;
            if (!computeReprojectionResidual(x_try, rt, obs1, obs2, &r_try)) {
                step *= 0.5;
                continue;
            }

            double try_cost = robustResidualCost(r_try, huber_delta_px);
            if (try_cost <= current_cost) {
                x = x_try;
                residual = r_try;
                current_cost = try_cost;
                accepted = true;
                break;
            }

            step *= 0.5;
        }

        if (!accepted || (step * dx).norm() < 1e-6)
            break;
    }

    return x;
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

Pose3D triangulatePoseWithRuntime(const Pose2D& left_pose,
                                  const Pose2D& right_pose,
                                  const StereoRuntime& rt,
                                  const TriangulationConfig& cfg) {
    Pose3D out;
    out.timestamp = left_pose.timestamp;

    const auto left_map = dedupeByBestConfidence(left_pose.joints);
    const auto right_map = dedupeByBestConfidence(right_pose.joints);

    std::vector<int> ids;
    ids.reserve(left_map.size());
    for (const auto& kv : left_map)
        ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());

    out.joints.reserve(ids.size());
    std::unordered_map<int, int> out_index;
    out_index.reserve(ids.size());

    std::vector<CandidateMatch> candidates;
    candidates.reserve(ids.size());

    for (int id : ids) {
        Joint3D joint;
        joint.id = id;
        const int idx = static_cast<int>(out.joints.size());
        out.joints.push_back(joint);
        out_index[id] = idx;

        auto r_it = right_map.find(id);
        if (r_it == right_map.end())
            continue;

        const Joint2D& left_joint = left_map.at(id);
        const Joint2D& right_joint = r_it->second;
        if (left_joint.confidence < cfg.min_joint_confidence ||
            right_joint.confidence < cfg.min_joint_confidence)
            continue;

        const double epi = epipolarDistancePx(rt.F, left_joint.uv, right_joint.uv);
        if (epi > cfg.max_epipolar_error_px)
            continue;

        std::vector<cv::Point2f> one_left{left_joint.uv};
        std::vector<cv::Point2f> one_right{right_joint.uv};
        std::vector<cv::Point2f> left_rect, right_rect;
        cv::undistortPoints(one_left, left_rect, rt.K1, rt.D1, rt.R1_rect, rt.P1_rect);
        cv::undistortPoints(one_right, right_rect, rt.K2, rt.D2, rt.R2_rect, rt.P2_rect);
        const double disparity = std::abs(left_rect[0].x - right_rect[0].x);
        if (disparity < cfg.min_disparity_px)
            continue;

        CandidateMatch cand;
        cand.id = id;
        cand.left_conf = left_joint.confidence;
        cand.right_conf = right_joint.confidence;
        cand.out_index = idx;
        cand.left_px = left_joint.uv;
        cand.right_px = right_joint.uv;
        candidates.push_back(cand);
    }

    if (candidates.empty())
        return out;

    std::vector<cv::Point2f> left_px;
    std::vector<cv::Point2f> right_px;
    left_px.reserve(candidates.size());
    right_px.reserve(candidates.size());
    for (const auto& c : candidates) {
        left_px.push_back(c.left_px);
        right_px.push_back(c.right_px);
    }

    std::vector<cv::Point2f> left_norm, right_norm;
    cv::undistortPoints(left_px, left_norm, rt.K1, rt.D1);
    cv::undistortPoints(right_px, right_norm, rt.K2, rt.D2);

    cv::Mat points4d;
    cv::triangulatePoints(rt.P1_norm, rt.P2_norm, left_norm, right_norm, points4d);
    cv::Mat points4d64;
    points4d.convertTo(points4d64, CV_64F);

    for (int i = 0; i < points4d64.cols; ++i) {
        const double w = points4d64.at<double>(3, i);
        if (std::abs(w) < kEps)
            continue;

        Eigen::Vector3d x(points4d64.at<double>(0, i) / w,
                          points4d64.at<double>(1, i) / w,
                          points4d64.at<double>(2, i) / w);

        if (!x.allFinite())
            continue;

        if (x.norm() > cfg.max_depth)
            continue;

        if (x.z() <= kEps)
            continue;

        const Eigen::Vector3d x2 = rt.R_e * x + rt.T_e;
        if (x2.z() <= kEps)
            continue;

        x = refineByReprojection(x, rt, candidates[i].left_px, candidates[i].right_px,
                                 cfg.max_reprojection_error_px);

        double mean_err = std::numeric_limits<double>::infinity();
        double max_err = std::numeric_limits<double>::infinity();
        if (!computeReprojectionResidual(x, rt, candidates[i].left_px, candidates[i].right_px,
                                         nullptr, &mean_err, &max_err))
            continue;

        if (max_err > cfg.max_reprojection_error_px)
            continue;

        Joint3D& out_joint = out.joints[candidates[i].out_index];
        out_joint.valid = true;
        out_joint.xyz = x;
        out_joint.reproj_err_px = mean_err;

        const float base_conf = std::sqrt(std::max(0.0f, candidates[i].left_conf * candidates[i].right_conf));
        const double quality = 1.0 - std::min(1.0, mean_err / std::max(kEps, cfg.max_reprojection_error_px));
        out_joint.confidence = static_cast<float>(base_conf * quality);
    }

    return out;
}

} // namespace

TriangulationResult triangulateFromImages(const std::string& image1_path,
                                          const std::string& image2_path,
                                          const TriangulationConfig& cfg) {
    // 1. Load images as grayscale
    cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);

    if (img1.empty())
        throw TriangulationError("Failed to load image: " + image1_path);
    if (img2.empty())
        throw TriangulationError("Failed to load image: " + image2_path);

    // 2. Detect SIFT keypoints and descriptors (sub-pixel accurate, float descriptors)
    auto sift = cv::SIFT::create(cfg.max_features);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    sift->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    sift->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    if (kp1.empty() || kp2.empty())
        throw TriangulationError("No keypoints detected in one or both images");

    // 3. Match with FLANN + Lowe's ratio test (FLANN is optimized for float descriptors)
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < cfg.match_ratio_thresh * m[1].distance)
            good_matches.push_back(m[0]);
    }

    std::cout << "Matches after ratio test: " << good_matches.size() << "\n";

    if (static_cast<int>(good_matches.size()) < cfg.min_matches)
        throw TriangulationError("Too few matches: " + std::to_string(good_matches.size())
                                 + " (need " + std::to_string(cfg.min_matches) + ")");

    // 4. Build default intrinsics
    double fx = static_cast<double>(img1.cols);
    double fy = fx;
    double cx = img1.cols / 2.0;
    double cy = img1.rows / 2.0;
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                            0, fy, cy,
                                            0,  0,  1);

    // Extract matched point coordinates
    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(good_matches.size());
    pts2.reserve(good_matches.size());
    for (const auto& m : good_matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    // 4b. Sub-pixel refinement on matched keypoint locations
    cv::TermCriteria subpix_criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
    cv::cornerSubPix(img1, pts1, cv::Size(5, 5), cv::Size(-1, -1), subpix_criteria);
    cv::cornerSubPix(img2, pts2, cv::Size(5, 5), cv::Size(-1, -1), subpix_criteria);

    // 5. Find Essential matrix
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999,
                                     cfg.max_reproj_error, inlier_mask);

    if (E.empty())
        throw TriangulationError("Failed to estimate Essential matrix (degenerate configuration)");

    int inlier_count = cv::countNonZero(inlier_mask);
    std::cout << "Inliers after Essential matrix: " << inlier_count << "\n";

    if (inlier_count < cfg.min_matches)
        throw TriangulationError("Too few inliers after Essential matrix estimation: "
                                 + std::to_string(inlier_count));

    // 6. Recover relative pose
    cv::Mat R, t;
    int pose_inliers = cv::recoverPose(E, pts1, pts2, K, R, t, inlier_mask);

    if (pose_inliers < cfg.min_matches)
        throw TriangulationError("Too few inliers after pose recovery: "
                                 + std::to_string(pose_inliers));

    // 7. Build projection matrices
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    K.copyTo(P1(cv::Rect(0, 0, 3, 3)));

    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P2 = K * Rt;

    // Collect inlier points
    std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
    for (int i = 0; i < static_cast<int>(pts1.size()); ++i) {
        if (inlier_mask.at<uchar>(i)) {
            inlier_pts1.push_back(pts1[i]);
            inlier_pts2.push_back(pts2[i]);
        }
    }

    // 8. Triangulate
    cv::Mat points4d;
    cv::triangulatePoints(P1, P2, inlier_pts1, inlier_pts2, points4d);

    // Build camera poses
    // Camera 1: identity (at origin)
    Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();

    // Camera 2: [R|t]
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            T2(r, c) = R.at<double>(r, c);
        T2(r, 3) = t.at<double>(r, 0);
    }

    Eigen::Vector3d cam1_pos = T1.block<3, 1>(0, 3);
    Eigen::Vector3d cam2_pos = T2.block<3, 1>(0, 3);

    // 9. Convert to Point3D, filtering invalid points
    TriangulationResult result;
    result.cameras.push_back({0.0, T1});
    result.cameras.push_back({1.0, T2});

    const Eigen::Vector3f cyan(0.2f, 0.8f, 1.0f);
    const Eigen::Vector3f line_color1(0.8f, 0.4f, 0.1f); // orange for cam1 rays
    const Eigen::Vector3f line_color2(0.1f, 0.4f, 0.8f); // blue for cam2 rays

    // First pass: collect valid 3D points with reprojection + parallax filtering
    const double min_parallax_deg = 1.0; // minimum angle between rays (degrees)
    const double min_parallax_cos = std::cos(min_parallax_deg * kPi / 180.0);

    for (int i = 0; i < points4d.cols; ++i) {
        float w = points4d.at<float>(3, i);
        if (std::abs(w) < 1e-6f)
            continue; // at infinity

        float x = points4d.at<float>(0, i) / w;
        float y = points4d.at<float>(1, i) / w;
        float z = points4d.at<float>(2, i) / w;

        if (z < 0.0f)
            continue; // behind camera

        Eigen::Vector3d pos(x, y, z);
        if (pos.norm() > cfg.max_depth)
            continue; // too far

        // Parallax angle check: angle between rays from each camera to the point.
        // Small angles mean depth is unreliable (ill-conditioned triangulation).
        Eigen::Vector3d ray1 = (pos - cam1_pos).normalized();
        Eigen::Vector3d ray2 = (pos - cam2_pos).normalized();
        double cos_angle = ray1.dot(ray2);
        if (cos_angle > min_parallax_cos)
            continue; // parallax too small, depth unreliable

        // Reprojection error check
        cv::Mat pt3d = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);

        cv::Mat reproj1 = P1 * pt3d;
        cv::Point2f rp1(reproj1.at<double>(0) / reproj1.at<double>(2),
                        reproj1.at<double>(1) / reproj1.at<double>(2));
        float err1 = cv::norm(rp1 - inlier_pts1[i]);

        cv::Mat reproj2 = P2 * pt3d;
        cv::Point2f rp2(reproj2.at<double>(0) / reproj2.at<double>(2),
                        reproj2.at<double>(1) / reproj2.at<double>(2));
        float err2 = cv::norm(rp2 - inlier_pts2[i]);

        if (err1 > cfg.max_reproj_error || err2 > cfg.max_reproj_error)
            continue; // poor triangulation

        result.cloud.push_back({pos, cyan});
    }

    // Second pass: add projection lines for a sparse subset only (~20 lines max)
    const int max_lines = 20;
    int n = static_cast<int>(result.cloud.size());
    int step = std::max(1, n / max_lines);
    for (int i = 0; i < n; i += step) {
        const auto& pos = result.cloud[i].position;
        result.projection_lines.push_back({cam1_pos, pos, line_color1});
        result.projection_lines.push_back({cam2_pos, pos, line_color2});
    }

    std::cout << "Reconstructed 3D points: " << result.cloud.size() << "\n";

    // 10. Optionally show 2D keypoint match visualization
    if (cfg.show_matches) {
        cv::Mat img1_color = cv::imread(image1_path, cv::IMREAD_COLOR);
        cv::Mat img2_color = cv::imread(image2_path, cv::IMREAD_COLOR);

        if (!img1_color.empty() && !img2_color.empty()) {
            // Collect all inlier keypoints
            std::vector<cv::KeyPoint> all_kp1, all_kp2;
            for (int i = 0; i < static_cast<int>(pts1.size()); ++i) {
                if (inlier_mask.at<uchar>(i)) {
                    all_kp1.push_back(cv::KeyPoint(pts1[i], 3.0f));
                    all_kp2.push_back(cv::KeyPoint(pts2[i], 3.0f));
                }
            }

            // Subsample: show at most ~60 evenly-spaced matches
            const int max_vis = 60;
            int total_inliers = static_cast<int>(all_kp1.size());
            int vis_step = std::max(1, total_inliers / max_vis);

            std::vector<cv::KeyPoint> vis_kp1, vis_kp2;
            std::vector<cv::DMatch> vis_matches;
            int idx = 0;
            for (int i = 0; i < total_inliers; i += vis_step) {
                vis_kp1.push_back(all_kp1[i]);
                vis_kp2.push_back(all_kp2[i]);
                vis_matches.push_back(cv::DMatch(idx, idx, 0));
                ++idx;
            }

            cv::Mat match_img;
            cv::drawMatches(img1_color, vis_kp1, img2_color, vis_kp2,
                            vis_matches, match_img,
                            cv::Scalar(0, 255, 200),   // match line color (cyan-green)
                            cv::Scalar(0, 140, 255),   // keypoint color (orange)
                            std::vector<char>(),
                            cv::DrawMatchesFlags::DEFAULT);

            // Resize if too large for screen
            const int max_width = 1600;
            if (match_img.cols > max_width) {
                double scale = static_cast<double>(max_width) / match_img.cols;
                cv::resize(match_img, match_img, cv::Size(), scale, scale);
            }

            cv::imshow("Stereo Matches (press any key to continue)", match_img);
            std::cout << "Showing " << vis_matches.size() << " of " << total_inliers
                      << " inlier matches. Press any key to launch 3D viewer...\n";
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    return result;
}

StereoCalibration loadStereoCalibration(const std::string& filepath) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw TriangulationError("Failed to open stereo calibration file: " + filepath);

    StereoCalibration calib;
    calib.K1 = readFirstMatrix(fs, {"K1", "cameraMatrix1", "M1"});
    calib.D1 = readFirstMatrix(fs, {"D1", "distCoeffs1"});
    calib.K2 = readFirstMatrix(fs, {"K2", "cameraMatrix2", "M2"});
    calib.D2 = readFirstMatrix(fs, {"D2", "distCoeffs2"});
    calib.R = readFirstMatrix(fs, {"R", "R12"});
    calib.T = readFirstMatrix(fs, {"T", "T12"});

    int width = 0;
    int height = 0;
    if (!fs["image_width"].empty())
        fs["image_width"] >> width;
    if (!fs["image_height"].empty())
        fs["image_height"] >> height;
    if ((width <= 0 || height <= 0) && !fs["imageSize"].empty()) {
        cv::FileNode image_size = fs["imageSize"];
        if (image_size.isSeq() && image_size.size() == 2) {
            width = static_cast<int>(image_size[0]);
            height = static_cast<int>(image_size[1]);
        } else {
            cv::Size s;
            image_size >> s;
            width = s.width;
            height = s.height;
        }
    }

    calib.image_size = cv::Size(width, height);

    // Validate eagerly so errors are reported close to config loading.
    static_cast<void>(buildRuntime(calib));
    return calib;
}

Pose2DSequence loadPose2DSequenceCSV(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open())
        throw TriangulationError("Failed to open keypoint CSV: " + filepath);

    struct Row {
        double ts = 0.0;
        int id = -1;
        float x = 0.0f;
        float y = 0.0f;
        float conf = 1.0f;
    };

    std::vector<Row> rows;
    std::string line;
    int line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;
        if (line.empty())
            continue;

        const auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
            line = line.substr(0, comment_pos);
        if (line.empty())
            continue;

        for (char& ch : line) {
            if (ch == ',')
                ch = ' ';
        }

        std::istringstream iss(line);
        Row row;
        if (!(iss >> row.ts >> row.id >> row.x >> row.y))
            throw TriangulationError("Invalid keypoint row in " + filepath + " at line " + std::to_string(line_no));
        if (!(iss >> row.conf))
            row.conf = 1.0f;

        rows.push_back(row);
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.ts == b.ts)
            return a.id < b.id;
        return a.ts < b.ts;
    });

    Pose2DSequence sequence;
    if (rows.empty())
        return sequence;

    Pose2D current;
    current.timestamp = rows.front().ts;
    for (const Row& row : rows) {
        if (std::abs(row.ts - current.timestamp) > 1e-9) {
            sequence.push_back(current);
            current = Pose2D();
            current.timestamp = row.ts;
        }

        Joint2D joint;
        joint.id = row.id;
        joint.uv = cv::Point2f(row.x, row.y);
        joint.confidence = row.conf;
        current.joints.push_back(joint);
    }
    sequence.push_back(current);

    return sequence;
}

Pose3D triangulatePoseFromStereoKeypoints(const Pose2D& left_pose,
                                          const Pose2D& right_pose,
                                          const StereoCalibration& calib,
                                          const TriangulationConfig& cfg) {
    StereoRuntime rt = buildRuntime(calib);
    return triangulatePoseWithRuntime(left_pose, right_pose, rt, cfg);
}

Pose3DSequence triangulatePoseSequenceFromStereoKeypoints(const Pose2DSequence& left_seq,
                                                          const Pose2DSequence& right_seq,
                                                          const StereoCalibration& calib,
                                                          const TriangulationConfig& cfg) {
    if (left_seq.size() != right_seq.size())
        throw TriangulationError("Left/right keypoint sequence size mismatch: "
                                 + std::to_string(left_seq.size()) + " vs " + std::to_string(right_seq.size()));

    StereoRuntime rt = buildRuntime(calib);

    Pose3DSequence output;
    output.reserve(left_seq.size());

    std::unordered_map<int, TrackState> tracks;
    std::vector<double> reproj_errors;
    reproj_errors.reserve(left_seq.size() * 20);
    int total_valid = 0;
    int total_joints = 0;

    const double alpha = std::clamp(cfg.temporal_smoothing_alpha, 0.0, 0.99);

    for (size_t i = 0; i < left_seq.size(); ++i) {
        Pose3D pose = triangulatePoseWithRuntime(left_seq[i], right_seq[i], rt, cfg);
        if (pose.timestamp == 0.0)
            pose.timestamp = left_seq[i].timestamp;

        for (auto& joint : pose.joints) {
            ++total_joints;
            TrackState& track = tracks[joint.id];

            if (joint.valid) {
                if (track.has_value) {
                    double dt = pose.timestamp - track.timestamp;
                    if (dt <= 0.0)
                        dt = 1.0 / 30.0;

                    Eigen::Vector3d filtered = alpha * track.xyz + (1.0 - alpha) * joint.xyz;
                    Eigen::Vector3d delta = filtered - track.xyz;
                    const double max_step = std::max(0.0, cfg.max_joint_speed_mps * dt);
                    if (delta.norm() > max_step && max_step > 0.0)
                        filtered = track.xyz + delta.normalized() * max_step;

                    joint.xyz = filtered;
                }

                track.has_value = true;
                track.xyz = joint.xyz;
                track.timestamp = pose.timestamp;
                track.confidence = joint.confidence;

                ++total_valid;
                if (std::isfinite(joint.reproj_err_px))
                    reproj_errors.push_back(joint.reproj_err_px);
                continue;
            }

            if (track.has_value) {
                const double gap = std::max(0.0, pose.timestamp - track.timestamp);
                if (gap <= cfg.hold_last_valid_seconds) {
                    joint.valid = true;
                    joint.xyz = track.xyz;
                    joint.confidence = std::max(0.05f, track.confidence * 0.35f);
                }
            }
        }

        output.push_back(std::move(pose));
    }

    if (cfg.verbose_pose_stats && !output.empty()) {
        const double valid_ratio = (total_joints > 0) ? (100.0 * total_valid / total_joints) : 0.0;
        const double avg_valid_per_frame = (output.empty()) ? 0.0 : static_cast<double>(total_valid) / output.size();
        const double med_reproj = median(reproj_errors);
        std::cout << "Pose sequence stats:\n"
                  << "  Frames: " << output.size() << "\n"
                  << "  Avg valid joints/frame: " << avg_valid_per_frame << "\n"
                  << "  Valid joint ratio: " << valid_ratio << "%\n"
                  << "  Median reprojection error: " << med_reproj << " px\n";
    }

    return output;
}

} // namespace vio
