#include "frontend/imu_tracking_pipeline.hpp"

#include "frontend/frame_pose_sync.hpp"
#include "io/image_sequence_reader.hpp"
#include "io/tracked_output_writer.hpp"

#include "imu/imu.hpp"
#include "tracking/lk_tracker.hpp"
#include "tracking/tracking_vis.hpp"
#include "tracking/feature_refresh.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdexcept>

void ImuTrackingPipeline::setImuCsvPath(const std::string& path)
{
    imu_csv_path_ = path;
}

void ImuTrackingPipeline::setImagesDir(const std::string& path)
{
    images_dir_ = path;
}

void ImuTrackingPipeline::setFrameTimestampsPath(const std::string& path)
{
    frame_timestamps_path_ = path;
}

void ImuTrackingPipeline::setOutputPosesCsv(const std::string& path)
{
    output_poses_csv_ = path;
}

void ImuTrackingPipeline::setOutputObservationsCsv(const std::string& path)
{
    output_observations_csv_ = path;
}

void ImuTrackingPipeline::setOutputVideoPath(const std::string& path)
{
    output_video_path_ = path;
}

void ImuTrackingPipeline::setGravity(const Eigen::Vector3d& gravity)
{
    gravity_ = gravity;
}

void ImuTrackingPipeline::setTrackingParams(
    int win_size,
    int max_level,
    int max_iters,
    float eps
)
{
    win_size_ = win_size;
    max_level_ = max_level;
    max_iters_ = max_iters;
    eps_ = eps;
}

void ImuTrackingPipeline::setRefreshParams(const FeatureRefreshParams& params)
{
    (void)params;
}

const std::vector<vio::TrackedFrame>& ImuTrackingPipeline::sequence() const
{
    return sequence_;
}

const std::vector<Pose>& ImuTrackingPipeline::imuTrajectory() const
{
    return imu_trajectory_;
}

bool ImuTrackingPipeline::loadInputs()
{
    image_paths_ = loadImagePaths(images_dir_, ".png");
    if (image_paths_.empty()) {
        std::cerr << "No images loaded from: " << images_dir_ << "\n";
        return false;
    }

    image_paths_ = loadImagePaths(images_dir_, ".png");
    if (image_paths_.empty()) {
        std::cerr << "No images loaded from: " << images_dir_ << "\n";
        return false;
    }

    frame_timestamps_ = loadImageTimestampsFromFilenames(image_paths_);
    if (frame_timestamps_.empty()) {
        std::cerr << "Failed to load timestamps from image filenames\n";
        return false;
    }


    if (image_paths_.size() != frame_timestamps_.size()) {
        std::cerr << "Image count and timestamp count do not match: "
                  << image_paths_.size() << " images vs "
                  << frame_timestamps_.size() << " timestamps\n";
        return false;
    }

    if (!loadImuCsv(imu_csv_path_, imu_samples_)) {
        std::cerr << "Failed to load IMU CSV: " << imu_csv_path_ << "\n";
        return false;
    }

    return true;
}

bool ImuTrackingPipeline::runImu()
{
    if (imu_samples_.empty()) {
        std::cerr << "IMU samples are empty\n";
        return false;
    }

    const double t0 = imu_samples_.front().t;
    const double t1 = imu_samples_.back().t;

    Pose pose;
    imu_trajectory_.clear();

    integrateImuFiltered(imu_samples_, t0, t1, pose, gravity_, imu_trajectory_
    );

    if (imu_trajectory_.empty()) {
        std::cerr << "IMU trajectory is empty after integration\n";
        return false;
    }

    return true;
}

void ImuTrackingPipeline::initializeTracks(const cv::Mat& first_gray)
{
    (void)first_gray;
}

void ImuTrackingPipeline::appendFrame(
    int frame_id,
    double timestamp,
    const std::vector<Track>& tracks
)
{
    vio::TrackedFrame frame;
    frame.state = buildFrameStateFromImu(frame_id, timestamp, imu_trajectory_);
    frame.observations.reserve(tracks.size());

    for (const auto& t : tracks) {
        vio::Observation obs;
        obs.frame_id = frame_id;
        obs.track_id = t.id;
        obs.uv = Eigen::Vector2d(t.pt.x, t.pt.y);
        obs.valid = true;
        frame.observations.push_back(obs);
    }

    sequence_.push_back(std::move(frame));
}

bool ImuTrackingPipeline::runTrackingAndSync()
{
    if (image_paths_.empty() || frame_timestamps_.empty()) {
        std::cerr << "Images or timestamps are empty\n";
        return false;
    }

    cv::Mat first_frame = cv::imread(image_paths_[0], cv::IMREAD_COLOR);
    if (first_frame.empty()) {
        std::cerr << "Failed to read first image: " << image_paths_[0] << "\n";
        return false;
    }

    const int width = first_frame.cols;
    const int height = first_frame.rows;

    cv::VideoWriter writer(
        output_video_path_,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        20.0,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << output_video_path_ << "\n";
        return false;
    }

    cv::Mat prev_gray;
    cv::cvtColor(first_frame, prev_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> initial_pts;
    cv::goodFeaturesToTrack(prev_gray, initial_pts, 100, 0.01, 10.0);

    if (initial_pts.empty()) {
        std::cerr << "No initial features found in first frame\n";
        return false;
    }

    std::vector<Track> tracks;
    int next_track_id = 0;

    for (const auto& p : initial_pts) {
        Track t;
        t.id = next_track_id++;
        t.pt = p;
        t.history.push_back(p);
        tracks.push_back(t);
    }

    FeatureRefreshParams refresh_params;
    refresh_params.minTrackedFeatures = 50;
    refresh_params.targetFeatures = 100;
    refresh_params.suppressionRadius = 10.0f;
    refresh_params.qualityLevel = 0.01;
    refresh_params.minDistance = 10.0;

    sequence_.clear();
    appendFrame(0, frame_timestamps_[0], tracks);

    writer.write(drawTrackingVisualization(first_frame, tracks));

    for (size_t img_idx = 1; img_idx < image_paths_.size(); ++img_idx) {
        cv::Mat curr_frame = cv::imread(image_paths_[img_idx], cv::IMREAD_COLOR);
        if (curr_frame.empty()) {
            std::cerr << "Failed to read image: " << image_paths_[img_idx] << "\n";
            continue;
        }

        cv::Mat curr_gray;
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> pts_prev;
        pts_prev.reserve(tracks.size());
        for (const auto& t : tracks) {
            pts_prev.push_back(t.pt);
        }

        std::vector<cv::Point2f> pts_curr;
        std::vector<uchar> status;
        std::vector<float> err;

        try {
            trackPointsPyramidalLK(
                prev_gray,
                curr_gray,
                pts_prev,
                pts_curr,
                status,
                err,
                win_size_,
                max_level_,
                max_iters_,
                eps_
            );
        } catch (const std::exception& e) {
            std::cerr << "Tracking failed on frame " << img_idx << ": " << e.what() << "\n";
            return false;
        }

        std::vector<Track> new_tracks;
        new_tracks.reserve(tracks.size());

        for (size_t i = 0; i < tracks.size(); ++i) {
            if (status[i]) {
                Track updated = tracks[i];
                updated.pt = pts_curr[i];
                updated.history.push_back(pts_curr[i]);
                new_tracks.push_back(updated);
            }
        }

        tracks = std::move(new_tracks);

        refreshTracksIfNeeded(curr_gray, tracks, next_track_id, refresh_params);

        appendFrame(static_cast<int>(img_idx), frame_timestamps_[img_idx], tracks);

        writer.write(drawTrackingVisualization(curr_frame, tracks, 15));

        prev_gray = curr_gray.clone();

        if (tracks.empty()) {
            std::cerr << "No tracks left on frame " << img_idx << "\n";
            break;
        }
    }

    writer.release();

    if (!writeFrameStatesCsv(output_poses_csv_, sequence_)) {
        std::cerr << "Failed to write poses CSV: " << output_poses_csv_ << "\n";
        return false;
    }

    if (!writeObservationsCsv(output_observations_csv_, sequence_)) {
        std::cerr << "Failed to write observations CSV: " << output_observations_csv_ << "\n";
        return false;
    }

    return true;
}

bool ImuTrackingPipeline::run()
{
    if (!loadInputs()) {
        return false;
    }

    if (!runImu()) {
        return false;
    }

    if (!runTrackingAndSync()) {
        return false;
    }

    return true;
}