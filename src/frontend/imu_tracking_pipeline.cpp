#include "frontend/imu_tracking_pipeline.hpp"

#include "io/image_sequence_reader.hpp"
#include "io/tracked_output_writer.hpp"

#include "tracking/tracking_vis.hpp"
#include "tracking/feature_refresh.hpp"

#include "io/landmark_output_writer.hpp"
#include "geometry/extrinsics.hpp"
#include "core/projection.hpp"
#include "keypoint_extraction/gaussian_blur.hpp"
#include "keypoint_extraction/shi_tomasi.hpp"
#include "keypoint_extraction/tpool_default.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <opencv2/opencv.hpp>

namespace {

float medianValue(std::vector<float> values)
{
    if (values.empty()) {
        return 0.0f;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

bool insideImage(const cv::Point2f& p, const cv::Size& size, float border)
{
    return std::isfinite(p.x) && std::isfinite(p.y) &&
           p.x >= border && p.y >= border &&
           p.x < static_cast<float>(size.width) - border &&
           p.y < static_cast<float>(size.height) - border;
}

cv::Mat buildSuppressionMask(
    const cv::Size& size,
    const std::vector<Track>& tracks,
    float radius)
{
    cv::Mat mask(size, CV_8U, cv::Scalar(255));
    const int r = std::max(2, static_cast<int>(std::round(radius)));
    for (const Track& t : tracks) {
        if (insideImage(t.pt, size, 1.0f)) {
            cv::circle(mask, t.pt, r, cv::Scalar(0), -1, cv::LINE_AA);
        }
    }
    return mask;
}

void topUpTracksWithGFTT(
    const cv::Mat& gray,
    std::vector<Track>& tracks,
    int& next_track_id,
    int target_features,
    float suppression_radius)
{
    if (static_cast<int>(tracks.size()) >= target_features) {
        return;
    }

    const int missing = target_features - static_cast<int>(tracks.size());
    cv::Mat mask = buildSuppressionMask(gray.size(), tracks, suppression_radius);

    std::vector<cv::Point2f> detected;
    cv::goodFeaturesToTrack(
        gray,
        detected,
        missing,
        0.008,
        7.0,
        mask,
        5,
        false,
        0.04
    );

    for (const cv::Point2f& p : detected) {
        Track t;
        t.id = next_track_id++;
        t.pt = p;
        t.history.push_back(p);
        tracks.push_back(std::move(t));
    }
}

} // namespace

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
    if (image_paths_.empty()) {
        std::cerr << "No images loaded from: " << images_dir_ << "\n";
        return false;
    }

    std::cout << "Loaded images: " << image_paths_.size() << "\n";

    frame_timestamps_.clear();

    if (!frame_timestamps_path_.empty()) {
        frame_timestamps_ = loadImageTimestampsFromFile(frame_timestamps_path_);
        if (!frame_timestamps_.empty()) {
            std::cout << "Loaded camera timestamps from file: "
                      << frame_timestamps_path_ << " ("
                      << frame_timestamps_.size() << ")\n";
        } else {
            std::cerr << "Could not load camera timestamps from file, falling back to image filenames: "
                      << frame_timestamps_path_ << "\n";
        }
    }

    if (frame_timestamps_.size() > image_paths_.size() * 2) {
        std::cerr << "Timestamp file has far more rows than images ("
                  << frame_timestamps_.size() << " vs " << image_paths_.size()
                  << "). This looks like an IMU CSV, so camera timestamps will be read from image filenames.\n";
        frame_timestamps_.clear();
    }

    if (frame_timestamps_.empty()) {
        frame_timestamps_ = loadImageTimestampsFromFilenames(image_paths_);
        if (!frame_timestamps_.empty()) {
            std::cout << "Loaded camera timestamps from image filenames: "
                      << frame_timestamps_.size() << "\n";
        }
    }

    if (frame_timestamps_.empty()) {
        std::cerr << "Failed to load camera timestamps from file or filenames\n";
        return false;
    }

    if (frame_timestamps_.size() < image_paths_.size()) {
        std::cerr << "Not enough timestamps for images: "
                  << frame_timestamps_.size() << " timestamps vs "
                  << image_paths_.size() << " images\n";
        return false;
    }

    if (frame_timestamps_.size() > image_paths_.size()) {
        std::cerr << "Timestamp file has extra rows: "
                  << frame_timestamps_.size() << " timestamps vs "
                  << image_paths_.size() << " images. Extra timestamps will be ignored.\n";
        frame_timestamps_.resize(image_paths_.size());
    }

    std::cout << "IMU loading skipped (visual-only mode).\n";
    return true;
}

bool ImuTrackingPipeline::runImu()
{
    std::cout << "IMU is temporarily disabled. Using visual-only pipeline.\n";
    imu_trajectory_.clear();
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

    const unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
    ThreadPool pool(static_cast<int>(numThreads));
    CustomShiTomasiDetector detector(pool, static_cast<int>(numThreads));

    ShiTomasiParams detectorParams;
    detectorParams.maxCorners = 350;
    detectorParams.qualityLevel = 0.008;
    detectorParams.minDistance = 7.0;
    detectorParams.blockSize = 5;
    detectorParams.gaussianSigma = 1.0;
    detectorParams.nmsRadius = 2;

    std::vector<cv::Point2f> initial_pts = detector.detect(first_gray, detectorParams);
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

    frame.state.frame_id = frame_id;
    frame.state.timestamp = timestamp;

    // Visual-only pose prior. It is intentionally smooth and small: enough for
    // two-view triangulation scale, but not allowed to create wild PnP jumps.
    const double x = 0.025 * static_cast<double>(sequence_.size());
    frame.state.t_wc = Eigen::Vector3d(x, 0.0, 0.0);
    frame.state.v_w = Eigen::Vector3d(0.025, 0.0, 0.0);
    frame.state.q_wc = Eigen::Quaterniond::Identity();

    frame.observations.reserve(tracks.size());

    for (const auto& t : tracks) {
        if (!std::isfinite(t.pt.x) || !std::isfinite(t.pt.y)) {
            continue;
        }

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
    if (sequence_.empty()) {
        std::cerr << "Sequence is empty, cannot triangulate landmarks\n";
        return false;
    }

    if (!camera_intrinsics_.isValid()) {
        std::cerr << "Camera intrinsics are invalid\n";
        return false;
    }

    landmarks_ = triangulateLandmarks(sequence_, camera_intrinsics_, triangulation_params_);
    std::cout << "Triangulated landmarks: " << landmarks_.size() << "\n";
    return true;
}

bool ImuTrackingPipeline::runTrackingAndSync()
{
    if (image_paths_.empty() || frame_timestamps_.empty()) {
        std::cerr << "Images or timestamps are empty\n";
        return false;
    }

    const size_t start_idx = 0;
    size_t end_idx = frame_timestamps_.size();
    const size_t max_frames = 260;
    if (max_frames > 0) {
        end_idx = std::min(end_idx, start_idx + max_frames);
    }

    cv::Mat first_frame = cv::imread(image_paths_[start_idx], cv::IMREAD_COLOR);
    if (first_frame.empty()) {
        std::cerr << "Failed to read first image: " << image_paths_[start_idx] << "\n";
        return false;
    }

    cv::VideoWriter writer(
        output_video_path_,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        20.0,
        first_frame.size()
    );

    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << output_video_path_ << "\n";
        return false;
    }

    cv::Mat prev_gray;
    cv::cvtColor(first_frame, prev_gray, cv::COLOR_BGR2GRAY);

    std::vector<Track> tracks;
    int next_track_id = 0;

    std::vector<cv::Point2f> initial_pts;
    cv::goodFeaturesToTrack(prev_gray, initial_pts, 350, 0.008, 7.0, cv::Mat(), 5, false, 0.04);
    if (initial_pts.empty()) {
        std::cerr << "No initial features found in first frame\n";
        return false;
    }

    for (const cv::Point2f& p : initial_pts) {
        Track t;
        t.id = next_track_id++;
        t.pt = p;
        t.history.push_back(p);
        tracks.push_back(std::move(t));
    }

    sequence_.clear();
    appendFrame(static_cast<int>(start_idx), frame_timestamps_[start_idx], tracks);
    writer.write(drawTrackingVisualization(first_frame, tracks, 25));

    const int target_features = 350;
    const int min_features_before_refresh = 180;
    const float suppression_radius = 7.0f;
    const float border = 4.0f;
    const float max_forward_backward_error = 1.5f;
    const float max_single_frame_motion = 60.0f;

    for (size_t img_idx = start_idx + 1; img_idx < end_idx; ++img_idx) {
        cv::Mat curr_frame = cv::imread(image_paths_[img_idx], cv::IMREAD_COLOR);
        if (curr_frame.empty()) {
            std::cerr << "Failed to read image: " << image_paths_[img_idx] << "\n";
            continue;
        }

        cv::Mat curr_gray;
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> pts_prev;
        pts_prev.reserve(tracks.size());
        for (const Track& t : tracks) {
            pts_prev.push_back(t.pt);
        }

        std::vector<cv::Point2f> pts_curr;
        std::vector<uchar> status_forward;
        std::vector<float> err_forward;

        if (!pts_prev.empty()) {
            cv::calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                pts_prev,
                pts_curr,
                status_forward,
                err_forward,
                cv::Size(std::max(9, win_size_), std::max(9, win_size_)),
                std::max(1, max_level_),
                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                 std::max(10, max_iters_), eps_)
            );
        }

        std::vector<cv::Point2f> pts_back;
        std::vector<uchar> status_backward;
        std::vector<float> err_backward;
        if (!pts_curr.empty()) {
            cv::calcOpticalFlowPyrLK(
                curr_gray,
                prev_gray,
                pts_curr,
                pts_back,
                status_backward,
                err_backward,
                cv::Size(std::max(9, win_size_), std::max(9, win_size_)),
                std::max(1, max_level_),
                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                 std::max(10, max_iters_), eps_)
            );
        }

        std::vector<float> motions;
        motions.reserve(tracks.size());
        std::vector<Track> kept_tracks;
        kept_tracks.reserve(tracks.size());

        for (size_t i = 0; i < tracks.size(); ++i) {
            if (i >= status_forward.size() || i >= status_backward.size() ||
                i >= pts_curr.size() || i >= pts_back.size()) {
                continue;
            }
            if (!status_forward[i] || !status_backward[i]) {
                continue;
            }

            const cv::Point2f& prev_pt = pts_prev[i];
            const cv::Point2f& curr_pt = pts_curr[i];
            const cv::Point2f& back_pt = pts_back[i];

            if (!insideImage(curr_pt, curr_gray.size(), border)) {
                continue;
            }

            const float fb_error = pointDistance(prev_pt, back_pt);
            const float motion = pointDistance(prev_pt, curr_pt);
            if (!std::isfinite(fb_error) || !std::isfinite(motion) ||
                fb_error > max_forward_backward_error ||
                motion > max_single_frame_motion) {
                continue;
            }

            Track updated = tracks[i];
            updated.pt = curr_pt;
            updated.history.push_back(curr_pt);
            if (updated.history.size() > 80) {
                updated.history.erase(updated.history.begin(), updated.history.end() - 80);
            }
            kept_tracks.push_back(std::move(updated));
            motions.push_back(motion);
        }

        tracks = std::move(kept_tracks);

        if (static_cast<int>(tracks.size()) < min_features_before_refresh) {
            topUpTracksWithGFTT(curr_gray, tracks, next_track_id, target_features, suppression_radius);
        }

        appendFrame(static_cast<int>(img_idx), frame_timestamps_[img_idx], tracks);
        writer.write(drawTrackingVisualization(curr_frame, tracks, 25));

        if ((img_idx - start_idx) % 25 == 0) {
            std::cout << "tracking frame " << img_idx
                      << ": active_tracks=" << tracks.size()
                      << ", median_flow_px=" << medianValue(motions) << "\n";
        }

        prev_gray = curr_gray.clone();

        if (tracks.empty()) {
            std::cerr << "No tracks left on frame " << img_idx << "\n";
            break;
        }
    }

    writer.release();
    return !sequence_.empty();
}

bool ImuTrackingPipeline::runTrackingWithImuPrior()
{
    std::cerr << "runTrackingWithImuPrior is disabled in visual-only mode.\n";
    return false;
}

bool ImuTrackingPipeline::runOnlyTrackingAndSync()
{
    sequence_.clear();
    imu_trajectory_.clear();
    landmarks_.clear();

    std::cout << "STEP 1: loadInputs\n";
    if (!loadInputs()) {
        return false;
    }

    std::cout << "STEP 2: runImu\n";
    if (!runImu()) {
        return false;
    }

    std::cout << "STEP 3: runTrackingAndSync\n";
    if (!runTrackingAndSync()) {
        return false;
    }

    return true;
}

bool ImuTrackingPipeline::run()
{
    return runOnlyTrackingAndSync();
}
