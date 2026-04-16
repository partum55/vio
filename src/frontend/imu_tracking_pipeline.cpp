#include "frontend/imu_tracking_pipeline.hpp"

#include "frontend/frame_pose_sync.hpp"
#include "io/image_sequence_reader.hpp"
#include "io/tracked_output_writer.hpp"

#include "imu/imu.hpp"
#include "tracking/lk_tracker.hpp"
#include "tracking/tracking_vis.hpp"
#include "tracking/feature_refresh.hpp"
#include "io/landmark_output_writer.hpp"
#include "triangulation/extrinsics.hpp"
#include "core/projection.hpp"
#include "keypoint_extraction/shi_tomasi.hpp"
#include "keypoint_extraction/tpool_default.hpp"

#include <unordered_map>

#include <opencv2/opencv.hpp>

#include <iostream>

namespace
{
    std::vector<Track> buildTracksFromFirstFrameSequence(
        const std::vector<vio::TrackedFrame>& sequence
    )
    {
        std::vector<Track> tracks;
        if (sequence.empty()) {
            return tracks;
        }

        const auto& first = sequence.front();
        tracks.reserve(first.observations.size());

        for (const auto& obs : first.observations) {
            if (!obs.valid) {
                continue;
            }

            Track t;
            t.id = obs.track_id;
            t.pt = cv::Point2f(
                static_cast<float>(obs.uv.x()),
                static_cast<float>(obs.uv.y())
            );
            t.history.push_back(t.pt);
            tracks.push_back(t);
        }

        return tracks;
    }
}

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

void ImuTrackingPipeline::setCameraIntrinsics(const CameraIntrinsics& intrinsics)
{
    camera_intrinsics_ = intrinsics;
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

void ImuTrackingPipeline::setCameraExtrinsics(const RigidTransform& T_bs)
{
    T_bs_ = T_bs;
}

bool ImuTrackingPipeline::loadInputs()
{
    image_paths_ = loadImagePaths(images_dir_, ".png");
    if (image_paths_.empty())
    {
        std::cerr << "No images loaded from: " << images_dir_ << "\n";
        return false;
    }

    std::cout << "Loaded images: " << image_paths_.size() << "\n";

    image_paths_ = loadImagePaths(images_dir_, ".png");
    if (image_paths_.empty())
    {
        std::cerr << "No images loaded from: " << images_dir_ << "\n";
        return false;
    }

    if (!frame_timestamps_path_.empty())
    {
        frame_timestamps_ = loadImageTimestampsFromFile(frame_timestamps_path_);
        if (frame_timestamps_.empty())
        {
            std::cerr << "Failed to load frame timestamps from file: "
                << frame_timestamps_path_ << "\n";
            return false;
        }
    }
    else
    {
        frame_timestamps_ = loadImageTimestampsFromFilenames(image_paths_);
        if (frame_timestamps_.empty())
        {
            std::cerr << "Failed to load timestamps from image filenames\n";
            return false;
        }
    }


    if (image_paths_.size() != frame_timestamps_.size())
    {
        std::cerr << "Image count and timestamp count do not match: "
            << image_paths_.size() << " images vs "
            << frame_timestamps_.size() << " timestamps\n";
        return false;
    }

    if (!loadImuCsv(imu_csv_path_, imu_samples_))
    {
        std::cerr << "Failed to load IMU CSV: " << imu_csv_path_ << "\n";
        return false;
    }

    return true;
}

bool ImuTrackingPipeline::runImu()
{
    if (imu_samples_.empty())
    {
        std::cerr << "IMU samples are empty\n";
        return false;
    }

    const double t0 = imu_samples_.front().t;
    const double t1 = imu_samples_.back().t;

    Pose pose;
    imu_trajectory_.clear();

    integrateImuFiltered(imu_samples_, t0, t1, pose, gravity_, imu_trajectory_
    );

    if (imu_trajectory_.empty())
    {
        std::cerr << "IMU trajectory is empty after integration\n";
        return false;
    }

    return true;
}

void ImuTrackingPipeline::initializeTracks(const cv::Mat& first_gray)
{
    if (first_gray.empty()) {
        throw std::runtime_error("initializeTracks: empty image");
    }

    if (first_gray.type() != CV_8UC1) {
        throw std::runtime_error("initializeTracks: expected CV_8UC1 grayscale image");
    }

    const unsigned int numThreads =
        std::max(1u, std::thread::hardware_concurrency());

    ThreadPool pool(static_cast<int>(numThreads));
    CustomShiTomasiDetector detector(pool, static_cast<int>(numThreads));

    ShiTomasiParams detectorParams;
    detectorParams.maxCorners = 100;
    detectorParams.qualityLevel = 0.01;
    detectorParams.minDistance = 10.0;
    detectorParams.blockSize = 5;
    detectorParams.gaussianSigma = 1.0;
    detectorParams.nmsRadius = 2;

    std::vector<cv::Point2f> initial_pts =
        detector.detect(first_gray, detectorParams);

    if (initial_pts.empty()) {
        throw std::runtime_error("initializeTracks: no initial features found");
    }
}

void ImuTrackingPipeline::setTriangulationParams(const TriangulationParams& params)
{
    triangulation_params_ = params;
}

const std::vector<vio::Landmark>& ImuTrackingPipeline::landmarks() const
{
    return landmarks_;
}

void ImuTrackingPipeline::setOutputLandmarksCsv(const std::string& path)
{
    output_landmarks_csv_ = path;
}

void ImuTrackingPipeline::appendFrame(
    int frame_id,
    double timestamp,
    const std::vector<Track>& tracks
)
{
    vio::TrackedFrame frame;

    frame.state = buildFrameStateFromImu(frame_id, timestamp, imu_trajectory_);

    const Eigen::Matrix3d R_wb = frame.state.q_wc.toRotationMatrix();
    const Eigen::Vector3d t_wb = frame.state.t_wc;

    frame.state.q_wc = Eigen::Quaterniond(R_wb);
    frame.state.q_wc.normalize();
    frame.state.t_wc = t_wb;

    frame.observations.reserve(tracks.size());

    for (const auto& t : tracks)
    {
        vio::Observation obs;
        obs.frame_id = frame_id;
        obs.track_id = t.id;
        obs.uv = Eigen::Vector2d(t.pt.x, t.pt.y);
        obs.valid = true;
        frame.observations.push_back(obs);
    }

    sequence_.push_back(std::move(frame));
}

bool ImuTrackingPipeline::runTriangulation()
{
    if (sequence_.empty())
    {
        std::cerr << "Sequence is empty, cannot triangulate landmarks\n";
        return false;
    }

    if (!camera_intrinsics_.isValid())
    {
        std::cerr << "Camera intrinsics are invalid\n";
        return false;
    }

    landmarks_ = triangulateLandmarks(
        sequence_,
        camera_intrinsics_,
        triangulation_params_
    );

    std::cout << "Triangulated landmarks: " << landmarks_.size() << "\n";
    return true;
}

bool ImuTrackingPipeline::runTrackingAndSync()
{
    if (image_paths_.empty() || frame_timestamps_.empty())
    {
        std::cerr << "Images or timestamps are empty\n";
        return false;
    }

    if (imu_trajectory_.empty())
    {
        std::cerr << "IMU trajectory is empty\n";
        return false;
    }

    const double t_min = imu_trajectory_.front().t;
    const double t_max = imu_trajectory_.back().t;

    size_t start_idx = 0;
    while (start_idx < frame_timestamps_.size() &&
        frame_timestamps_[start_idx] < t_min)
    {
        ++start_idx;
    }

    size_t end_idx = frame_timestamps_.size();
    while (end_idx > start_idx &&
        frame_timestamps_[end_idx - 1] > t_max)
    {
        --end_idx;
    }

    if (start_idx >= end_idx)
    {
        std::cerr << "No overlapping time range between frames and IMU\n";
        return false;
    }
    const size_t max_frames = 220; // 0 без обмеження

    if (max_frames > 0) {
        end_idx = std::min(end_idx, start_idx + max_frames);
    }

    cv::Mat first_frame = cv::imread(image_paths_[start_idx], cv::IMREAD_COLOR);

    if (first_frame.empty())
    {
        std::cerr << "Failed to read first image: " << image_paths_[start_idx] << "\n";
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

    if (!writer.isOpened())
    {
        std::cerr << "Failed to open output video: " << output_video_path_ << "\n";
        return false;
    }

    cv::Mat prev_gray, prev_gray_f, prev_blur;

    cv::cvtColor(first_frame, prev_gray, cv::COLOR_BGR2GRAY);
    prev_gray.convertTo(prev_gray_f, CV_32F, 1.0 / 255.0);
    prev_blur = gaussianBlurCustomFloat(prev_gray_f);

    std::vector<cv::Point2f> initial_pts;
    cv::goodFeaturesToTrack(prev_gray, initial_pts, 100, 0.01, 10.0);

    if (initial_pts.empty())
    {
        std::cerr << "No initial features found in first frame\n";
        return false;
    }

    std::vector<Track> tracks;
    int next_track_id = 0;

    for (const auto& p : initial_pts)
    {
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

    appendFrame(
        static_cast<int>(start_idx),
        frame_timestamps_[start_idx],
        tracks
    );

    writer.write(drawTrackingVisualization(first_frame, tracks));

    for (size_t img_idx = start_idx + 1; img_idx < end_idx; ++img_idx)
    {
        cv::Mat curr_frame = cv::imread(image_paths_[img_idx], cv::IMREAD_COLOR);
        if (curr_frame.empty())
        {
            std::cerr << "Failed to read image: " << image_paths_[img_idx] << "\n";
            continue;
        }

        cv::Mat curr_gray, curr_gray_f, curr_blur;

        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
        curr_gray.convertTo(curr_gray_f, CV_32F, 1.0 / 255.0);
        curr_blur = gaussianBlurCustomFloat(curr_gray_f);

        std::vector<cv::Point2f> pts_prev;
        pts_prev.reserve(tracks.size());
        for (const auto& t : tracks)
        {
            pts_prev.push_back(t.pt);
        }

        std::vector<cv::Point2f> pts_curr;
        std::vector<uchar> status;
        std::vector<float> err;

        try
        {
            trackPointsPyramidalLK(
                prev_blur,
                curr_blur,
                pts_prev,
                pts_curr,
                status,
                err,
                win_size_,
                max_level_,
                max_iters_,
                eps_
            );
        }
        catch (const std::exception& e)
        {
            std::cerr << "Tracking failed on frame " << img_idx << ": " << e.what() << "\n";
            return false;
        }

        std::vector<Track> new_tracks;
        new_tracks.reserve(tracks.size());

        for (size_t i = 0; i < tracks.size(); ++i)
        {
            if (status[i])
            {
                Track updated = tracks[i];
                updated.pt = pts_curr[i];
                updated.history.push_back(pts_curr[i]);
                new_tracks.push_back(updated);
            }
        }

        tracks = std::move(new_tracks);

        refreshTracksIfNeeded(curr_gray, tracks, next_track_id, refresh_params);

        appendFrame(
            static_cast<int>(img_idx),
            frame_timestamps_[img_idx],
            tracks
        );

        writer.write(drawTrackingVisualization(curr_frame, tracks, 15));

        prev_blur = curr_blur.clone();

        if (tracks.empty())
        {
            std::cerr << "No tracks left on frame " << img_idx << "\n";
            break;
        }
    }

    writer.release();

    return true;
}

bool ImuTrackingPipeline::runTrackingWithImuPrior()
{
    if (image_paths_.empty() || frame_timestamps_.empty()) {
        std::cerr << "Images or timestamps are empty\n";
        return false;
    }

    if (imu_trajectory_.empty()) {
        std::cerr << "IMU trajectory is empty\n";
        return false;
    }

    if (landmarks_.empty()) {
        std::cerr << "Landmarks are empty, cannot run IMU-prior tracking\n";
        return false;
    }

    if (sequence_.empty()) {
        std::cerr << "Sequence is empty, cannot run IMU-prior tracking\n";
        return false;
    }

    std::unordered_map<int, Eigen::Vector3d> landmarks_by_track;
    for (const auto& lm : landmarks_) {
        if (lm.valid) {
            landmarks_by_track[lm.track_id] = lm.p_w;
        }
    }

    const double t_min = imu_trajectory_.front().t;
    const double t_max = imu_trajectory_.back().t;

    size_t start_idx = 0;
    while (start_idx < frame_timestamps_.size() &&
           frame_timestamps_[start_idx] < t_min) {
        ++start_idx;
    }

    size_t end_idx = frame_timestamps_.size();
    while (end_idx > start_idx &&
           frame_timestamps_[end_idx - 1] > t_max) {
        --end_idx;
    }

    if (start_idx >= end_idx) {
        std::cerr << "No overlapping time range between frames and IMU\n";
        return false;
    }

    const size_t max_frames = 220; // 0 без обмеження

    if (max_frames > 0) {
        end_idx = std::min(end_idx, start_idx + max_frames);
    }

    cv::Mat first_frame = cv::imread(image_paths_[start_idx], cv::IMREAD_COLOR);
    if (first_frame.empty()) {
        std::cerr << "Failed to read first image: " << image_paths_[start_idx] << "\n";
        return false;
    }

    const int width = first_frame.cols;
    const int height = first_frame.rows;

    cv::VideoWriter writer(
        "imu_prior_" + output_video_path_,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        20.0,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Failed to open IMU-prior output video\n";
        return false;
    }

    cv::Mat prev_gray, prev_gray_f, prev_blur;

    cv::cvtColor(first_frame, prev_gray, cv::COLOR_BGR2GRAY);
    prev_gray.convertTo(prev_gray_f, CV_32F, 1.0 / 255.0);
    prev_blur = gaussianBlurCustomFloat(prev_gray_f);

    std::vector<Track> tracks = buildTracksFromFirstFrameSequence(sequence_);
    if (tracks.empty()) {
        std::cerr << "Failed to initialize IMU-prior tracks from sequence\n";
        return false;
    }

    int next_track_id = 0;
    for (const auto& t : tracks) {
        next_track_id = std::max(next_track_id, t.id + 1);
    }

    FeatureRefreshParams refresh_params;
    refresh_params.minTrackedFeatures = 50;
    refresh_params.targetFeatures = 100;
    refresh_params.suppressionRadius = 10.0f;
    refresh_params.qualityLevel = 0.01;
    refresh_params.minDistance = 10.0;

    std::vector<vio::TrackedFrame> imu_prior_sequence;
    imu_prior_sequence.clear();

    appendFrame(
        static_cast<int>(start_idx),
        frame_timestamps_[start_idx],
        tracks
    );
    imu_prior_sequence.push_back(sequence_.back());
    sequence_.pop_back();

    writer.write(drawTrackingVisualization(first_frame, tracks));

    for (size_t img_idx = start_idx + 1; img_idx < end_idx; ++img_idx) {
        cv::Mat curr_frame = cv::imread(image_paths_[img_idx], cv::IMREAD_COLOR);
        if (curr_frame.empty()) {
            std::cerr << "Failed to read image: " << image_paths_[img_idx] << "\n";
            continue;
        }

        cv::Mat curr_gray, curr_gray_f, curr_blur;

        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
        curr_gray.convertTo(curr_gray_f, CV_32F, 1.0 / 255.0);
        curr_blur = gaussianBlurCustomFloat(curr_gray_f);

        std::vector<cv::Point2f> pts_prev;
        std::vector<cv::Point2f> pts_guess;

        pts_prev.reserve(tracks.size());
        pts_guess.reserve(tracks.size());

        const vio::FrameState curr_state = buildFrameStateFromImu(
            static_cast<int>(img_idx),
            frame_timestamps_[img_idx],
            imu_trajectory_
        );

        for (const auto& t : tracks) {
            pts_prev.push_back(t.pt);

            auto it = landmarks_by_track.find(t.id);
            if (it != landmarks_by_track.end()) {
                cv::Point2f uv_pred;

                if (projectLandmarkToFrame(curr_state, camera_intrinsics_, it->second, uv_pred)) {
                    const float max_jump = 10.0f;

                    const bool inside_image =
                        uv_pred.x >= 0.0f && uv_pred.x < static_cast<float>(curr_gray.cols) &&
                        uv_pred.y >= 0.0f && uv_pred.y < static_cast<float>(curr_gray.rows);

                    const bool reasonable_jump =
                        cv::norm(uv_pred - t.pt) < max_jump;

                    if (inside_image && reasonable_jump) {
                        pts_guess.push_back(uv_pred);
                    } else {
                        pts_guess.push_back(t.pt);
                    }
                } else {
                    pts_guess.push_back(t.pt);
                }
            } else {
                pts_guess.push_back(t.pt);
            }
        }

        std::vector<cv::Point2f> pts_curr;
        std::vector<uchar> status;
        std::vector<float> err;

        try {
            trackPointsPyramidalLKWithGuess(
                prev_blur,
                curr_blur,
                pts_prev,
                pts_guess,
                pts_curr,
                status,
                err,
                win_size_,
                max_level_,
                max_iters_,
                eps_
            );
        } catch (const std::exception& e) {
            std::cerr << "IMU-prior tracking failed on frame " << img_idx
                      << ": " << e.what() << "\n";
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

        appendFrame(
            static_cast<int>(img_idx),
            frame_timestamps_[img_idx],
            tracks
        );
        imu_prior_sequence.push_back(sequence_.back());
        sequence_.pop_back();

        writer.write(drawTrackingVisualization(curr_frame, tracks, 15));

        prev_blur = curr_blur.clone();

        if (tracks.empty()) {
            std::cerr << "No tracks left in IMU-prior tracking on frame " << img_idx << "\n";
            break;
        }
    }

    writer.release();

    sequence_ = std::move(imu_prior_sequence);
    return true;
}

bool ImuTrackingPipeline::run()
{
    sequence_.clear();
    imu_trajectory_.clear();
    landmarks_.clear();

    std::cout << "STEP 1: loadInputs\n";
    if (!loadInputs())
    {
        return false;
    }

    std::cout << "STEP 2: runImu\n";
    if (!runImu())
    {
        return false;
    }

    std::cout << "STEP 3: runTrackingAndSync\n";
    if (!runTrackingAndSync())
    {
        return false;
    }

    std::cout << "STEP 4: runTriangulation\n";
    if (!runTriangulation())
    {
        return false;
    }

    std::cout << "STEP 5: runTrackingWithImuPrior\n";
    if (!runTrackingWithImuPrior())
    {
        return false;
    }

    if (!writeFrameStatesCsv(output_poses_csv_, sequence_))
    {
        std::cerr << "Failed to write poses CSV: " << output_poses_csv_ << "\n";
        return false;
    }

    if (!writeObservationsCsv(output_observations_csv_, sequence_))
    {
        std::cerr << "Failed to write observations CSV: " << output_observations_csv_ << "\n";
        return false;
    }

    if (!writeLandmarksCsv(output_landmarks_csv_, landmarks_))
    {
        std::cerr << "Failed to write landmarks CSV: " << output_landmarks_csv_ << "\n";
        return false;
    }

    return true;
}
