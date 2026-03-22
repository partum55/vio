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
#include <limits>
#include <numeric>
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
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(12, 12, 18));
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

} // namespace vio
