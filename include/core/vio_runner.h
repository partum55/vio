#pragma once

#include "core/data_generator.h"
#include "core/dataset.h"
#include "core/rerun_stream.h"
#include "core/types.h"

#include <filesystem>
#include <string>

namespace vio {

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

RunResult runVisualInertialOdometry(const Dataset& dataset,
                                    const RunConfig& config,
                                    RerunStreamClient* stream_client);

RunResult runSyntheticDemo(const GeneratorConfig& generator_config,
                           const RunConfig& config,
                           RerunStreamClient* stream_client);

} // namespace vio
