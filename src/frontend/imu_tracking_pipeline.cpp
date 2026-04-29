#include "frontend/imu_tracking_pipeline.hpp"

#include "core/dataset_streamer.h"
#include "frontend/frame_pose_sync.hpp"
#include "io/image_sequence_reader.hpp"
#include "io/tracked_output_writer.hpp"

#include "imu/imu.hpp"
#include "tracking/tracking_vis.hpp"
#include "io/landmark_output_writer.hpp"
#include "geometry/geometry_backend.hpp"
#include "pipeline/vio_pipeline.hpp"
#include "geometry/extrinsics.hpp"   // якщо RigidTransform тепер тут

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>

#include <opencv2/opencv.hpp>

#include <iostream>

namespace
{
    std::int64_t timestampToNs(double timestamp_s)
    {
        return static_cast<std::int64_t>(std::llround(timestamp_s * 1e9));
    }

    vio::Dataset buildStreamingDataset(
        const std::vector<std::string>& image_paths,
        const std::vector<double>& frame_timestamps,
        const std::vector<ImuSample>& imu_samples,
        size_t start_idx,
        size_t end_idx
    )
    {
        vio::Dataset dataset;
        dataset.imu_samples = imu_samples;

        if (start_idx < image_paths.size()) {
            dataset.root = std::filesystem::path(image_paths[start_idx]).parent_path();
        }

        dataset.frames.reserve(end_idx - start_idx);
        for (size_t i = start_idx; i < end_idx; ++i) {
            vio::DatasetFrame frame;
            frame.timestamp_s = frame_timestamps[i];
            frame.timestamp_ns = timestampToNs(frame.timestamp_s);
            frame.frame_index = i;
            frame.image_path = image_paths[i];
            dataset.frames.push_back(std::move(frame));
        }

        return dataset;
    }

    void drainStreamedImuSamples(
        ThreadSafeQueue<ImuSample>& queue,
        std::optional<ImuSample>& lookahead,
        double timestamp_s
    )
    {
        if (lookahead) {
            if (lookahead->t <= timestamp_s) {
                lookahead.reset();
            } else {
                return;
            }
        }

        while (true) {
            auto sample = queue.try_deque();
            if (!sample) {
                break;
            }
            if (sample->t > timestamp_s) {
                lookahead = std::move(*sample);
                break;
            }
        }
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
    VisualFrontendParams params;

    params.tracker.winSize = win_size;
    params.tracker.maxLevel = max_level;
    params.tracker.maxIters = max_iters;
    params.tracker.eps = eps;

    params.initialFeatures = 100;
    params.minTrackedFeatures = 50;

    params.refresh.minTrackedFeatures = 50;
    params.refresh.targetFeatures = 100;
    params.refresh.suppressionRadius = 10.0f;
    params.refresh.qualityLevel = 0.01;
    params.refresh.minDistance = 10.0;

    frontend_.setParams(params);
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

bool ImuTrackingPipeline::runTrackingAndSync()
{
    if (image_paths_.empty() || frame_timestamps_.empty()) {
        std::cerr << "Images or timestamps are empty\n";
        return false;
    }

    if (imu_trajectory_.empty()) {
        std::cerr << "IMU trajectory is empty\n";
        return false;
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
        std::cerr << "No overlapping time range\n";
        return false;
    }

	vio::GeometryBackend geometry(camera_intrinsics_);
	vio::VioPipeline vio_pipeline(geometry);

    vio::Dataset streaming_dataset = buildStreamingDataset(
        image_paths_,
        frame_timestamps_,
        imu_samples_,
        start_idx,
        end_idx
    );
    vio::DatasetStreamer streamer(streaming_dataset);
    streamer.start();
    std::optional<ImuSample> imu_lookahead;

    auto first_cf = streamer.imgQueue().deque();
    if (!first_cf || first_cf->frame_index != start_idx) {
        std::cerr << "Failed to read first frame\n";
        streamer.stop();
        return false;
    }

    cv::Mat first_frame = std::move(first_cf->image);
    drainStreamedImuSamples(streamer.imuQueue(), imu_lookahead, first_cf->timestamp_s);
    if (first_frame.empty()) {
        std::cerr << "Failed to read first frame\n";
        streamer.stop();
        return false;
    }

    cv::VideoWriter writer(
        output_video_path_,
        cv::VideoWriter::fourcc('m','p','4','v'),
        20.0,
        cv::Size(first_frame.cols, first_frame.rows)
    );

    sequence_.clear();

    // === INIT PIVOT ===
    vio::FrameState first_state = buildFrameStateFromImu(
        static_cast<int>(first_cf->frame_index),
        first_cf->timestamp_s,
        imu_trajectory_
    );

	std::cout << "first frame " << first_state.frame_id
          << " t_wc = " << first_state.t_wc.transpose()
          << "\n";

    frontend_.setPivot(
        static_cast<int>(first_cf->frame_index),
        first_cf->timestamp_s,
        first_frame,
        first_state
    );

    auto first_output = frontend_.track(
        static_cast<int>(first_cf->frame_index),
        first_cf->timestamp_s,
        first_frame,
        first_state
    );

	vio_pipeline.processFrame(first_output.frame);
    writer.write(drawTrackingVisualization(first_frame, first_output.tracks));

    // === MAIN LOOP ===
    while (auto cf = streamer.imgQueue().deque()) {
        cv::Mat frame = std::move(cf->image);
        const int frame_id = static_cast<int>(cf->frame_index);
        const double timestamp = cf->timestamp_s;
        drainStreamedImuSamples(streamer.imuQueue(), imu_lookahead, timestamp);

        vio::FrameState state = buildFrameStateFromImu(
            frame_id,
            timestamp,
            imu_trajectory_
        );

		std::cout << "frame " << frame_id
          << " t_wc = " << state.t_wc.transpose()
          << "\n";

        auto output = frontend_.track(
            frame_id,
            timestamp,
            frame,
            state
        );

		vio_pipeline.processFrame(output.frame);
        writer.write(drawTrackingVisualization(frame, output.tracks, 15));

        // === RESET PIVOT ===
        if (!output.enough_tracks) {
            frontend_.setPivot(
                frame_id,
                timestamp,
                frame,
                state
            );
        }
    }

	sequence_ = vio_pipeline.frames();
	landmarks_ = vio_pipeline.landmarks().getValidLandmarks();
    writer.release();
	streamer.stop();
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
