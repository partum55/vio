#pragma once

#include "core/data_generator.h"
#include "core/dataset.h"
#include "core/rerun_stream.h"
#include "core/types.h"

#include <filesystem>
#include <string>

namespace vio {

class DatasetStreamer;

struct RunConfig {
    std::string host = "127.0.0.1";
    int port = 9877;
    bool stream_to_rerun = true;
    std::filesystem::path video_output = "first_view.mp4";
    bool write_video = true;
};

struct RunResult {
    Trajectory trajectory;
    std::size_t processed_frames = 0;
    std::size_t streamed_frames = 0;
    std::filesystem::path video_path;
};

struct ViconReplayConfig {
    double playback_rate = 1.0;
    double visual_scale = 4.0;
    double first_view_fps = 20.0;
    int first_view_width = 1280;
    int first_view_height = 720;
    double first_view_fx = 700.0;
    double first_view_fy = 700.0;
};

RunResult runVisualInertialOdometry(const Dataset& dataset,
                                    const RunConfig& config,
                                    RerunStreamClient* stream_client);

// Queue-based overload: streamer supplies pre-loaded frames and IMU samples.
// Calls streamer.start() internally; caller must not start it beforehand.
RunResult runVisualInertialOdometry(DatasetStreamer& streamer,
                                    const Dataset& dataset,
                                    const RunConfig& config,
                                    RerunStreamClient* stream_client);

RunResult runSyntheticDemo(const GeneratorConfig& generator_config,
                           const RunConfig& config,
                           RerunStreamClient* stream_client);

RunResult runViconLiveDemo(const std::filesystem::path& dataset_root,
                           const ViconReplayConfig& replay_config,
                           const RunConfig& config,
                           RerunStreamClient* stream_client);

} // namespace vio
