#include "core/vio_runner.h"

#include "core/data_generator.h"
#include "imu/imu.h"
#include "keypoints/shi_tomasi.hpp"
#include "keypoints/tpool_default.hpp"
#include "tracking/klt_tracker.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace vio {
namespace {

double clampMagnitude(Eigen::Vector3d& vec, double max_norm) {
    const double norm = vec.norm();
    if (norm > max_norm && norm > 1e-9) {
        vec *= max_norm / norm;
    }
    return vec.norm();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    if ((values.size() % 2U) != 0U) {
        return *mid;
    }
    const double hi = *mid;
    std::nth_element(values.begin(), mid - 1, values.end());
    return 0.5 * (hi + *(mid - 1));
}

std::string trimCopy(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool nextCsvField(std::stringstream& ss, std::string& field) {
    if (!std::getline(ss, field, ',')) {
        return false;
    }
    field = trimCopy(field);
    return true;
}

bool parseViconRow(const std::string& line,
                   double& timestamp_s,
                   Eigen::Vector3d& position,
                   Eigen::Quaterniond& orientation) {
    if (line.empty() || line[0] == '#') {
        return false;
    }

    std::stringstream ss(line);
    std::string ts_field;
    std::string px_field;
    std::string py_field;
    std::string pz_field;
    std::string qw_field;
    std::string qx_field;
    std::string qy_field;
    std::string qz_field;
    if (!nextCsvField(ss, ts_field) ||
        !nextCsvField(ss, px_field) ||
        !nextCsvField(ss, py_field) ||
        !nextCsvField(ss, pz_field) ||
        !nextCsvField(ss, qw_field) ||
        !nextCsvField(ss, qx_field) ||
        !nextCsvField(ss, qy_field) ||
        !nextCsvField(ss, qz_field)) {
        throw std::runtime_error("malformed Vicon row: " + line);
    }

    timestamp_s = static_cast<double>(std::stoll(ts_field)) * 1e-9;
    position = Eigen::Vector3d(std::stod(px_field), std::stod(py_field), std::stod(pz_field));
    orientation = Eigen::Quaterniond(
        std::stod(qw_field),
        std::stod(qx_field),
        std::stod(qy_field),
        std::stod(qz_field));
    orientation.normalize();
    return true;
}

struct ViconReplaySample {
    double timestamp_s = 0.0;
    Eigen::Vector3d raw_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d scaled_position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond raw_orientation = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond view_orientation = Eigen::Quaterniond::Identity();
};

Eigen::Quaterniond makeViewOrientation(const Eigen::Vector3d& forward_hint,
                                       const Eigen::Vector3d& world_up = Eigen::Vector3d::UnitZ()) {
    Eigen::Vector3d forward = forward_hint;
    if (forward.norm() < 1e-6) {
        forward = Eigen::Vector3d(1.0, 0.0, -0.15);
    }
    forward.normalize();

    Eigen::Vector3d up = world_up.normalized();
    if (std::abs(forward.dot(up)) > 0.95) {
        up = Eigen::Vector3d::UnitY();
    }

    Eigen::Vector3d right = forward.cross(up);
    if (right.norm() < 1e-6) {
        right = Eigen::Vector3d::UnitX();
    }
    right.normalize();
    const Eigen::Vector3d cam_up = right.cross(forward).normalized();

    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    rotation.col(0) = right;
    rotation.col(1) = cam_up;
    rotation.col(2) = -forward;
    return Eigen::Quaterniond(rotation).normalized();
}

std::vector<ViconReplaySample> loadViconReplaySamples(const std::filesystem::path& csv_path,
                                                      double visual_scale) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open Vicon csv: " + csv_path.string());
    }

    std::vector<ViconReplaySample> samples;
    std::string line;
    while (std::getline(file, line)) {
        double timestamp_s = 0.0;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
        if (!parseViconRow(line, timestamp_s, position, orientation)) {
            continue;
        }

        ViconReplaySample sample;
        sample.timestamp_s = timestamp_s;
        sample.raw_position = position;
        sample.scaled_position = position * visual_scale;
        sample.raw_orientation = orientation;
        samples.push_back(sample);
    }

    if (samples.empty()) {
        throw std::runtime_error("no Vicon samples found in " + csv_path.string());
    }

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const std::size_t prev_idx = i == 0 ? i : i - 1;
        const std::size_t next_idx = i + 1 < samples.size() ? i + 1 : i;

        Eigen::Vector3d motion = samples[next_idx].scaled_position - samples[prev_idx].scaled_position;
        if (motion.norm() < 1e-6) {
            motion = samples[i].raw_orientation * Eigen::Vector3d::UnitX();
        }
        motion.z() -= 0.15 * std::max(1.0, motion.norm());
        samples[i].view_orientation = makeViewOrientation(motion);
    }

    return samples;
}

void addSegmentPoints(PointCloud& cloud,
                      const Eigen::Vector3d& start,
                      const Eigen::Vector3d& end,
                      const Eigen::Vector3f& color,
                      double spacing = 0.2) {
    const double length = (end - start).norm();
    const int count = std::max(2, static_cast<int>(std::ceil(length / std::max(1e-3, spacing))) + 1);
    for (int i = 0; i < count; ++i) {
        const double t = count == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(count - 1);
        Point3D pt;
        pt.position = start + t * (end - start);
        pt.color = color;
        cloud.push_back(pt);
    }
}

void addGate(PointCloud& cloud,
             const Eigen::Vector3d& center,
             const Eigen::Vector3d& forward_hint,
             double width,
             double height,
             const Eigen::Vector3f& color) {
    Eigen::Vector3d forward_xy = forward_hint;
    forward_xy.z() = 0.0;
    if (forward_xy.norm() < 1e-6) {
        forward_xy = Eigen::Vector3d::UnitX();
    }
    forward_xy.normalize();

    Eigen::Vector3d right(-forward_xy.y(), forward_xy.x(), 0.0);
    if (right.norm() < 1e-6) {
        right = Eigen::Vector3d::UnitY();
    }
    right.normalize();
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();

    const Eigen::Vector3d top_left = center - right * (0.5 * width) + up * (0.5 * height);
    const Eigen::Vector3d top_right = center + right * (0.5 * width) + up * (0.5 * height);
    const Eigen::Vector3d bottom_left = center - right * (0.5 * width) - up * (0.5 * height);
    const Eigen::Vector3d bottom_right = center + right * (0.5 * width) - up * (0.5 * height);

    addSegmentPoints(cloud, top_left, top_right, color, 0.16);
    addSegmentPoints(cloud, top_right, bottom_right, color, 0.16);
    addSegmentPoints(cloud, bottom_right, bottom_left, color, 0.16);
    addSegmentPoints(cloud, bottom_left, top_left, color, 0.16);
}

PointCloud buildViconEnvironment(const std::vector<ViconReplaySample>& samples,
                                 double visual_scale,
                                 double ground_z) {
    PointCloud cloud;

    const double corridor_drop = 0.35 * visual_scale;
    const double stripe_half_width = 2.4 * visual_scale;
    const double stripe_spacing = 0.18 * visual_scale;
    const double centerline_spacing = 0.28 * visual_scale;
    const double tether_spacing = 0.22 * visual_scale;
    const double gate_forward_offset = 4.0 * visual_scale;
    const double gate_width = 6.0 * visual_scale;
    const double gate_height = 4.0 * visual_scale;
    const double gate_tether_spacing = 0.25 * visual_scale;

    const Eigen::Vector3f ground_color(0.96f, 0.56f, 0.24f);
    const Eigen::Vector3f center_color(1.00f, 0.84f, 0.32f);
    const Eigen::Vector3f tether_color(0.82f, 0.71f, 0.55f);
    const Eigen::Vector3f gate_a(0.93f, 0.30f, 0.22f);
    const Eigen::Vector3f gate_b(0.86f, 0.28f, 0.62f);

    for (std::size_t i = 0; i < samples.size(); i += 18) {
        const std::size_t next_idx = std::min(samples.size() - 1, i + 18);
        Eigen::Vector3d ground_here = samples[i].scaled_position;
        ground_here.z() = std::max(ground_z, samples[i].scaled_position.z() - corridor_drop);
        Eigen::Vector3d ground_next = samples[next_idx].scaled_position;
        ground_next.z() = std::max(ground_z, samples[next_idx].scaled_position.z() - corridor_drop);
        addSegmentPoints(cloud, ground_here, ground_next, center_color, centerline_spacing);

        Eigen::Vector3d motion = ground_next - ground_here;
        if (motion.head<2>().norm() < 1e-6) {
            motion = Eigen::Vector3d::UnitX();
        }
        Eigen::Vector3d right(-motion.y(), motion.x(), 0.0);
        if (right.norm() < 1e-6) {
            right = Eigen::Vector3d::UnitY();
        }
        right.normalize();

        const Eigen::Vector3d stripe_left = ground_here - right * stripe_half_width;
        const Eigen::Vector3d stripe_right = ground_here + right * stripe_half_width;
        addSegmentPoints(cloud, stripe_left, stripe_right, ground_color, stripe_spacing);

        if ((i / 18) % 4 == 0) {
            addSegmentPoints(cloud, ground_here, samples[i].scaled_position, tether_color, tether_spacing);
        }
    }

    for (std::size_t i = 90; i < samples.size(); i += 220) {
        const std::size_t next_idx = std::min(samples.size() - 1, i + 30);
        Eigen::Vector3d forward = samples[next_idx].scaled_position - samples[i].scaled_position;
        forward.z() = 0.0;
        if (forward.norm() < 1e-6) {
            forward = Eigen::Vector3d::UnitX();
        }
        forward.normalize();

        Eigen::Vector3d gate_center = samples[i].scaled_position + forward * gate_forward_offset;
        gate_center.z() = samples[i].scaled_position.z();
        addGate(
            cloud,
            gate_center,
            forward,
            gate_width,
            gate_height,
            ((i / 220) % 2 == 0) ? gate_a : gate_b);
        addSegmentPoints(
            cloud,
            Eigen::Vector3d(
                gate_center.x(),
                gate_center.y(),
                std::max(ground_z, samples[i].scaled_position.z() - corridor_drop)),
            gate_center,
            Eigen::Vector3f(0.45f, 0.45f, 0.50f),
            gate_tether_spacing);
    }

    return cloud;
}

cv::Mat renderSyntheticFrame(const PointCloud& cloud,
                             const CameraPose& pose,
                             int width,
                             int height,
                             double fx,
                             double fy,
                             double cx,
                             double cy);

cv::Mat renderViconFirstView(const PointCloud& environment,
                             const CameraPose& pose,
                             int width,
                             int height,
                             double fx,
                             double fy) {
    cv::Mat frame = renderSyntheticFrame(environment, pose, width, height, fx, fy, width * 0.5, height * 0.5);

    cv::putText(
        frame,
        "Vicon demo first_view",
        cv::Point(32, 48),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(220, 240, 255),
        2,
        cv::LINE_AA);

    // HUD: position and heading
    const Eigen::Vector3d pos     = pose.T_wc.block<3, 1>(0, 3);
    const Eigen::Matrix3d rot     = pose.T_wc.block<3, 3>(0, 0);
    const Eigen::Vector3d cam_fwd = rot.col(2);
    const Eigen::Vector3d cam_up  = rot.col(1);

    const double heading_deg = std::fmod(
        std::atan2(cam_fwd.y(), cam_fwd.x()) * (180.0 / M_PI) + 360.0, 360.0);
    const double roll_rad = std::atan2(cam_up.x(), cam_up.z());

    char hud[128];
    std::snprintf(hud, sizeof(hud), "pos: %.2f  %.2f  %.2f", pos.x(), pos.y(), pos.z());
    cv::putText(frame, hud,
        cv::Point(16, height - 48),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 230, 255), 1, cv::LINE_AA);

    std::snprintf(hud, sizeof(hud), "hdg: %.0f deg", heading_deg);
    cv::putText(frame, hud,
        cv::Point(16, height - 22),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 230, 255), 1, cv::LINE_AA);

    // Artificial horizon
    const int cx = width / 2;
    const int cy = height / 2;
    const int hl = 56;
    const int dx = static_cast<int>(hl * std::cos(roll_rad));
    const int dy = static_cast<int>(hl * std::sin(roll_rad));
    cv::line(frame,
        cv::Point(cx - dx, cy + dy), cv::Point(cx + dx, cy - dy),
        cv::Scalar(80, 220, 255), 2, cv::LINE_AA);
    cv::circle(frame, cv::Point(cx, cy), 3, cv::Scalar(80, 220, 255), cv::FILLED, cv::LINE_AA);
    cv::line(frame,
        cv::Point(cx, cy - 14), cv::Point(cx, cy + 14),
        cv::Scalar(80, 220, 255), 1, cv::LINE_AA);

    return frame;
}

std::vector<ImuSample> collectImuSlice(const std::vector<ImuSample>& imu,
                                       std::size_t& imu_index,
                                       double t0,
                                       double t1) {
    std::vector<ImuSample> slice;

    while (imu_index < imu.size() && imu[imu_index].t < t0) {
        ++imu_index;
    }

    std::size_t probe = imu_index;
    while (probe < imu.size() && imu[probe].t <= t1) {
        slice.push_back(imu[probe]);
        ++probe;
    }
    imu_index = probe;
    return slice;
}

CameraPose makePose(double timestamp_s,
                    const Eigen::Vector3d& position,
                    const Eigen::Quaterniond& orientation) {
    CameraPose pose;
    pose.timestamp = timestamp_s;
    pose.T_wc = Eigen::Matrix4d::Identity();
    pose.T_wc.block<3, 3>(0, 0) = orientation.normalized().toRotationMatrix();
    pose.T_wc.block<3, 1>(0, 3) = position;
    return pose;
}

cv::Mat renderSyntheticFrame(const PointCloud& cloud,
                             const CameraPose& pose,
                             int width,
                             int height,
                             double fx,
                             double fy,
                             double cx,
                             double cy) {
    cv::Mat image(height, width, CV_8UC3);
    const cv::Vec3d sky_top(232.0, 188.0, 118.0);
    const cv::Vec3d sky_bottom(198.0, 150.0, 94.0);
    const cv::Vec3d ground_top(112.0, 126.0, 148.0);
    const cv::Vec3d ground_bottom(62.0, 78.0, 108.0);
    const int horizon = height * 9 / 16;
    for (int y = 0; y < height; ++y) {
        const bool is_sky = y < horizon;
        const double t = is_sky
            ? static_cast<double>(y) / std::max(1, horizon)
            : static_cast<double>(y - horizon) / std::max(1, height - horizon - 1);
        const cv::Vec3d color = is_sky
            ? ((1.0 - t) * sky_top + t * sky_bottom)
            : ((1.0 - t) * ground_top + t * ground_bottom);
        cv::line(
            image,
            cv::Point(0, y),
            cv::Point(width - 1, y),
            cv::Scalar(color[0], color[1], color[2]),
            1,
            cv::LINE_8);
    }
    cv::line(
        image,
        cv::Point(0, horizon),
        cv::Point(width - 1, horizon),
        cv::Scalar(180, 210, 240),
        2,
        cv::LINE_AA);
    const Eigen::Matrix4d T_cw = pose.T_wc.inverse();

    for (const auto& point : cloud) {
        const Eigen::Vector4d p_w(point.position.x(), point.position.y(), point.position.z(), 1.0);
        const Eigen::Vector4d p_c_h = T_cw * p_w;
        const Eigen::Vector3d p_c = p_c_h.head<3>();
        if (p_c.z() <= 0.05) {
            continue;
        }

        const int u = static_cast<int>(std::lround(fx * (p_c.x() / p_c.z()) + cx));
        const int v = static_cast<int>(std::lround(fy * (p_c.y() / p_c.z()) + cy));
        if (u < 0 || v < 0 || u >= width || v >= height) {
            continue;
        }

        const cv::Scalar color(
            std::clamp(static_cast<int>(point.color.z() * 255.0f), 0, 255),
            std::clamp(static_cast<int>(point.color.y() * 255.0f), 0, 255),
            std::clamp(static_cast<int>(point.color.x() * 255.0f), 0, 255));
        const int radius = std::max(1, static_cast<int>(4.0 / std::max(1.0, p_c.z())));
        cv::circle(image, cv::Point(u, v), radius, color, cv::FILLED, cv::LINE_AA);
    }

    return image;
}

} // namespace

RunResult runVisualInertialOdometry(const Dataset& dataset,
                                    const RunConfig& config,
                                    RerunStreamClient* stream_client) {
    if (dataset.frames.empty()) {
        throw std::runtime_error("dataset has no camera frames");
    }

    RunResult result;
    result.video_path = config.write_video ? config.video_output : std::filesystem::path{};

    cv::VideoWriter writer;
    if (config.write_video) {
        writer.open(
            config.video_output.string(),
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            20.0,
            cv::Size(dataset.camera.width, dataset.camera.height));
        if (!writer.isOpened()) {
            throw std::runtime_error("failed to open video writer for " + config.video_output.string());
        }
    }

    const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
    ThreadPool pool(static_cast<int>(hw_threads));
    CustomShiTomasiDetector detector(pool, static_cast<int>(hw_threads));

    ShiTomasiParams detector_params;
    detector_params.maxCorners = 600;
    detector_params.minDistance = 12.0;
    detector_params.qualityLevel = 0.02;
    detector_params.blockSize = 5;
    detector_params.gaussianSigma = 1.0;
    detector_params.nmsRadius = 2;

    const Eigen::Vector3d gravity(0.0, 0.0, 9.81);

    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_points;

    Eigen::Vector3d fused_position = Eigen::Vector3d::Zero();
    ImuPose imu_pose;
    imu_pose.t = dataset.frames.front().timestamp_s;
    ImuPose prev_imu_pose = imu_pose;
    bool have_previous_frame = false;
    std::size_t imu_index = 0;

    if (stream_client && stream_client->isConnected()) {
        stream_client->sendInit(dataset);
    }

    for (const DatasetFrame& frame : dataset.frames) {
        cv::Mat image = cv::imread(frame.image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("failed to read image: " + frame.image_path.string());
        }
        if (image.cols != dataset.camera.width || image.rows != dataset.camera.height) {
            cv::resize(image, image, cv::Size(dataset.camera.width, dataset.camera.height));
        }

        if (config.write_video) {
            writer.write(image);
        }

        const cv::Mat gray = toGrayU8(image);
        std::vector<cv::Point2f> curr_points;
        std::size_t valid_tracks = 0;
        Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc = Eigen::Vector3d::Zero();

        if (!have_previous_frame) {
            prev_points = detector.detect(gray, detector_params);
        } else {
            std::vector<cv::Point2f> tracked_points;
            std::vector<uchar> status;
            std::vector<float> err;
            trackPoints(prev_gray, gray, prev_points, tracked_points, status, err, 9, 3, 12, 1e-3f);

            std::vector<cv::Point2f> filtered_prev;
            std::vector<cv::Point2f> filtered_curr;
            filtered_prev.reserve(tracked_points.size());
            filtered_curr.reserve(tracked_points.size());

            std::vector<double> dx_values;
            std::vector<double> dy_values;
            std::vector<double> flow_values;

            for (std::size_t i = 0; i < tracked_points.size(); ++i) {
                if (!status[i]) {
                    continue;
                }
                const cv::Point2f& p0 = prev_points[i];
                const cv::Point2f& p1 = tracked_points[i];
                if (p1.x < 0.0f || p1.y < 0.0f ||
                    p1.x >= static_cast<float>(gray.cols) ||
                    p1.y >= static_cast<float>(gray.rows)) {
                    continue;
                }
                filtered_prev.push_back(p0);
                filtered_curr.push_back(p1);
                dx_values.push_back(static_cast<double>(p1.x - p0.x));
                dy_values.push_back(static_cast<double>(p1.y - p0.y));
                flow_values.push_back(std::hypot(dx_values.back(), dy_values.back()));
            }

            valid_tracks = filtered_curr.size();
            curr_points = filtered_curr;

            const std::vector<ImuSample> imu_slice = collectImuSlice(
                dataset.imu_samples,
                imu_index,
                prev_imu_pose.t,
                frame.timestamp_s);

            if (!imu_slice.empty()) {
                for (const ImuSample& sample : imu_slice) {
                    gyro += sample.gyro;
                    acc += sample.acc;
                }
                gyro /= static_cast<double>(imu_slice.size());
                acc /= static_cast<double>(imu_slice.size());

                std::vector<ImuPose> interval_traj;
                integrateImuFiltered(
                    imu_slice,
                    prev_imu_pose.t,
                    frame.timestamp_s,
                    imu_pose,
                    gravity,
                    interval_traj);
            } else {
                imu_pose.t = frame.timestamp_s;
            }

            const Eigen::Vector3d inertial_step = imu_pose.p - prev_imu_pose.p;
            const double dt = std::max(1e-3, frame.timestamp_s - prev_imu_pose.t);

            const double median_dx = median(dx_values);
            const double median_dy = median(dy_values);
            const double median_flow = median(flow_values);

            Eigen::Vector3d visual_step_camera(
                -median_dx / std::max(1.0, dataset.camera.fx),
                -median_dy / std::max(1.0, dataset.camera.fy),
                0.02 + 0.35 * median_flow / std::max(1.0, 0.5 * (dataset.camera.fx + dataset.camera.fy)));

            visual_step_camera.x() *= 0.25;
            visual_step_camera.y() *= 0.25;
            Eigen::Vector3d visual_step_world = imu_pose.q * visual_step_camera;

            Eigen::Vector3d fused_step = 0.8 * inertial_step + 0.2 * visual_step_world;
            const double max_step = 3.0 * dt + 0.03;
            clampMagnitude(fused_step, max_step);

            fused_position += fused_step;
            prev_imu_pose = imu_pose;

            if (valid_tracks < 80) {
                curr_points = detector.detect(gray, detector_params);
            }
        }

        if (!have_previous_frame) {
            prev_imu_pose.t = frame.timestamp_s;
            imu_pose.t = frame.timestamp_s;
            curr_points = prev_points;
            have_previous_frame = true;
        }

        const Eigen::Quaterniond orientation = imu_pose.q.normalized();
        result.trajectory.push_back(makePose(frame.timestamp_s, fused_position, orientation));
        ++result.processed_frames;

        if (stream_client && stream_client->isConnected()) {
            StreamSample sample;
            sample.timestamp_s = frame.timestamp_s;
            sample.position = fused_position;
            sample.orientation = orientation;
            sample.gyro = gyro;
            sample.acc = acc;
            sample.image_path = std::filesystem::absolute(frame.image_path);
            sample.track_count = valid_tracks;
            if (stream_client->sendSample(sample)) {
                ++result.streamed_frames;
            }
        }

        prev_gray = gray;
        prev_points = curr_points.empty() ? detector.detect(gray, detector_params) : curr_points;
    }

    if (stream_client && stream_client->isConnected()) {
        stream_client->sendDone(result.processed_frames);
    }

    return result;
}

RunResult runSyntheticDemo(const GeneratorConfig& generator_config,
                           const RunConfig& config,
                           RerunStreamClient* stream_client) {
    const Trajectory trajectory = generateTrajectory(generator_config);
    const PointCloud cloud = generatePointCloud(generator_config);

    const int width = 1280;
    const int height = 720;
    const double fx = 700.0;
    const double fy = 700.0;
    const double cx = width * 0.5;
    const double cy = height * 0.5;

    RunResult result;
    result.video_path = config.write_video ? config.video_output : std::filesystem::path{};
    result.trajectory = trajectory;

    cv::VideoWriter writer;
    if (config.write_video) {
        writer.open(
            config.video_output.string(),
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            30.0,
            cv::Size(width, height));
        if (!writer.isOpened()) {
            throw std::runtime_error("failed to open video writer for " + config.video_output.string());
        }
    }

    if (stream_client && stream_client->isConnected()) {
        stream_client->sendSyntheticInit();
        stream_client->sendPointCloud(cloud);
    }

    std::filesystem::path live_frame_dir;
    if (stream_client && stream_client->isConnected()) {
        live_frame_dir = std::filesystem::temp_directory_path() / "vio_demo_frames";
        std::filesystem::create_directories(live_frame_dir);
    }

    Eigen::Vector3d prev_pos = trajectory.front().T_wc.block<3, 1>(0, 3);
    for (std::size_t i = 0; i < trajectory.size(); ++i) {
        const CameraPose& pose = trajectory[i];
        const Eigen::Vector3d pos = pose.T_wc.block<3, 1>(0, 3);
        const Eigen::Quaterniond q(pose.T_wc.block<3, 3>(0, 0));
        const cv::Mat frame = renderSyntheticFrame(cloud, pose, width, height, fx, fy, cx, cy);

        if (config.write_video) {
            writer.write(frame);
        }

        StreamSample sample;
        sample.timestamp_s = pose.timestamp;
        sample.position = pos;
        sample.orientation = q.normalized();
        sample.gyro = Eigen::Vector3d::Zero();
        sample.acc = Eigen::Vector3d::Zero();
        if (i != 0) {
            sample.acc = (pos - prev_pos) * 30.0;
        }
        sample.track_count = cloud.size();
        if (!live_frame_dir.empty()) {
            const std::filesystem::path frame_path =
                live_frame_dir / ("frame_" + std::to_string(i) + ".png");
            cv::imwrite(frame_path.string(), frame);
            sample.image_path = frame_path;
        }
        if (stream_client && stream_client->isConnected() && stream_client->sendSample(sample)) {
            ++result.streamed_frames;
        }

        prev_pos = pos;
        ++result.processed_frames;
    }

    if (stream_client && stream_client->isConnected()) {
        stream_client->sendDone(result.processed_frames);
    }

    return result;
}

RunResult runViconLiveDemo(const std::filesystem::path& dataset_root,
                           const ViconReplayConfig& replay_config,
                           const RunConfig& config,
                           RerunStreamClient* stream_client) {
    if (replay_config.playback_rate <= 0.0) {
        throw std::runtime_error("playback rate must be positive");
    }
    if (replay_config.visual_scale <= 0.0) {
        throw std::runtime_error("visual scale must be positive");
    }
    if (replay_config.first_view_fps <= 0.0) {
        throw std::runtime_error("first_view_fps must be positive");
    }

    const std::filesystem::path vicon_csv = dataset_root / "vicon0" / "data.csv";
    const std::vector<ViconReplaySample> samples = loadViconReplaySamples(vicon_csv, replay_config.visual_scale);
    double min_scaled_z = samples.front().scaled_position.z();
    double sum_scaled_z = 0.0;
    for (const auto& sample : samples) {
        min_scaled_z = std::min(min_scaled_z, sample.scaled_position.z());
        sum_scaled_z += sample.scaled_position.z();
    }
    const double mean_scaled_z = sum_scaled_z / static_cast<double>(samples.size());
    const double ground_z = std::max(
        min_scaled_z - 0.25 * replay_config.visual_scale,
        mean_scaled_z - 0.75 * replay_config.visual_scale);

    const PointCloud environment = buildViconEnvironment(samples, replay_config.visual_scale, ground_z);

    RunResult result;
    result.video_path = std::filesystem::path{};

    std::filesystem::path live_frame_dir;
    std::filesystem::path latest_frame_path;
    if (stream_client && stream_client->isConnected()) {
        live_frame_dir = std::filesystem::temp_directory_path() / "vio_vicon_frames";
        std::filesystem::create_directories(live_frame_dir);
    }

    if (stream_client && stream_client->isConnected()) {
        stream_client->sendInit(
            (dataset_root / "vicon0").string(),
            replay_config.first_view_width,
            replay_config.first_view_height,
            replay_config.first_view_fx,
            replay_config.first_view_fy,
            replay_config.first_view_width * 0.5,
            replay_config.first_view_height * 0.5,
            replay_config.visual_scale,
            ground_z);
    }

    double previous_timestamp_s = samples.front().timestamp_s;
    double last_render_timestamp_s = -std::numeric_limits<double>::infinity();
    std::size_t render_count = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const ViconReplaySample& sample_data = samples[i];
        if (i != 0) {
            const double dt_s = std::max(0.0, (sample_data.timestamp_s - previous_timestamp_s) / replay_config.playback_rate);
            if (dt_s > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(dt_s));
            }
        }

        result.trajectory.push_back(makePose(sample_data.timestamp_s, sample_data.raw_position, sample_data.raw_orientation));
        ++result.processed_frames;

        if (stream_client && stream_client->isConnected()) {
            if (!live_frame_dir.empty() &&
                (latest_frame_path.empty() ||
                 sample_data.timestamp_s - last_render_timestamp_s >= (1.0 / replay_config.first_view_fps))) {
                const CameraPose view_pose = makePose(
                    sample_data.timestamp_s,
                    sample_data.scaled_position,
                    sample_data.view_orientation);
                cv::Mat frame = renderViconFirstView(
                    environment,
                    view_pose,
                    replay_config.first_view_width,
                    replay_config.first_view_height,
                    replay_config.first_view_fx,
                    replay_config.first_view_fy);
                latest_frame_path = live_frame_dir / ("frame_" + std::to_string(render_count % 2) + ".jpg");
                cv::imwrite(
                    latest_frame_path.string(),
                    frame,
                    std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 85});
                ++render_count;
                last_render_timestamp_s = sample_data.timestamp_s;
            }

            StreamSample sample;
            sample.timestamp_s = sample_data.timestamp_s;
            sample.position = sample_data.scaled_position;
            sample.orientation = sample_data.view_orientation;
            sample.track_count = environment.size();
            sample.image_path = latest_frame_path;
            if (stream_client->sendSample(sample)) {
                ++result.streamed_frames;
            }
        }

        previous_timestamp_s = sample_data.timestamp_s;
    }

    if (stream_client && stream_client->isConnected()) {
        stream_client->sendDone(result.processed_frames);
    }

    (void)config;
    return result;
}

} // namespace vio
