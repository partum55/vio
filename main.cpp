#include "pipeline/vio_pipeline.hpp"
#include "visualization/points_video_renderer.hpp"
#include "visualization/rerun_visualizer.hpp"

#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct AppOptions {
    bool help = false;
    bool points_video = false;
    bool rerun = true;
    bool realtime = false;
    double stream_rate = 1.0;
    std::size_t max_image_queue = 8;

    std::string video_out = "../results/imu_tracking_visualization.mp4";
    std::optional<std::filesystem::path> dataset_root;
};

void printUsage(const char* executable)
{
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "       " << executable << " <dataset_root> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --points, --points-video       Write MP4 visualization with tracked 2D points.\n"
        << "  --video-out <path>             Output path for --points-video.\n"
        << "  --no-rerun                     Disable Rerun live visualization logger.\n"
        << "  --realtime                     Replay dataset using timestamp-based timing.\n"
        << "  --rate <value>                 Playback rate for --realtime, default 1.0.\n"
        << "  --max-image-queue <count>      Bounded streamer image queue size, default 8.\n"
        << "  --help                         Show this help.\n";
}

std::string requireValue(int& index, int argc, char** argv, std::string_view option)
{
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string(option) + " requires a value");
    }
    ++index;
    return argv[index];
}

AppOptions parseArgs(int argc, char** argv)
{
    AppOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--points" || arg == "--points-video") {
            options.points_video = true;
        } else if (arg == "--video-out") {
            options.video_out = requireValue(i, argc, argv, arg);
        } else if (arg == "--no-rerun") {
            options.rerun = false;
        } else if (arg == "--realtime") {
            options.realtime = true;
        } else if (arg == "--rate") {
            options.stream_rate = std::stod(requireValue(i, argc, argv, arg));
            if (options.stream_rate <= 0.0) {
                throw std::runtime_error("--rate must be positive");
            }
        } else if (arg == "--max-image-queue") {
            options.max_image_queue =
                static_cast<std::size_t>(std::stoull(requireValue(i, argc, argv, arg)));
            if (options.max_image_queue == 0) {
                throw std::runtime_error("--max-image-queue must be greater than zero");
            }
        } else if (!arg.empty() && arg.front() != '-') {
            if (options.dataset_root.has_value()) {
                throw std::runtime_error("only one dataset_root positional argument is supported");
            }
            options.dataset_root = std::filesystem::path(std::string(arg));
        } else {
            throw std::runtime_error("unknown option: " + std::string(arg));
        }
    }
    return options;
}

class PointsVideoLogger {
public:
    explicit PointsVideoLogger(std::filesystem::path output_path)
        : output_path_(std::move(output_path))
    {
    }

    void log(
        const vio::TrackedFrame& frame,
        const std::vector<vio::Track>& tracks,
        const std::vector<vio::Landmark>& landmarks,
        vio::VioStatus status,
        const std::filesystem::path& image_path)
    {
        (void)frame;
        (void)landmarks;
        (void)status;

        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "warning: could not read image for points video: "
                      << image_path << "\n";
            return;
        }

        openIfNeeded(image.size());
        writer_.write(vio::drawPointsVideoFrame(image, tracks, 15));
    }

private:
    void openIfNeeded(const cv::Size& frame_size)
    {
        if (writer_.isOpened()) {
            return;
        }

        const std::filesystem::path parent = output_path_.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        if (output_path_.extension().empty()) {
            output_path_ += ".mp4";
        }

        struct CodecCandidate {
            const char* name;
            int fourcc;
        };
        const std::array<CodecCandidate, 4> codecs{{
            {"avc1", cv::VideoWriter::fourcc('a', 'v', 'c', '1')},
            {"H264", cv::VideoWriter::fourcc('H', '2', '6', '4')},
            {"x264", cv::VideoWriter::fourcc('x', '2', '6', '4')},
            {"mp4v", cv::VideoWriter::fourcc('m', 'p', '4', 'v')},
        }};

        for (const CodecCandidate& codec : codecs) {
            writer_.open(
                output_path_.string(),
                codec.fourcc,
                20.0,
                frame_size,
                true);
            if (writer_.isOpened()) {
                std::cout << "Points video codec: " << codec.name << "\n";
                return;
            }
        }

        if (!writer_.isOpened()) {
            throw std::runtime_error(
                "failed to open MP4 points video writer: " + output_path_.string());
        }
    }

    std::filesystem::path output_path_;
    cv::VideoWriter writer_;
};

} // namespace

int main(int argc, char** argv)
{
    AppOptions options;
    try {
        options = parseArgs(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n\n";
        printUsage(argv[0]);
        return 2;
    }

    if (options.help) {
        printUsage(argv[0]);
        return 0;
    }

    vio::VioRunConfig config;
    config.imu_csv_path = "../data/imu0/data.csv";
    config.images_dir = "../data/cam0/undistorted_alpha0/";
    // The images in this dataset are timestamp-named and aligned with imu0/data.csv.
    // data/frame_timestamps.txt belongs to a different time range and breaks pose propagation.
    config.frame_timestamps_path.clear();

    if (options.dataset_root.has_value()) {
        const std::filesystem::path root = *options.dataset_root;
        config.imu_csv_path = (root / "imu0" / "data.csv").string();

        const std::filesystem::path undistorted_images = root / "cam0" / "undistorted_alpha0";
        const std::filesystem::path raw_images = root / "cam0" / "data";
        config.images_dir =
            std::filesystem::is_directory(undistorted_images)
                ? undistorted_images.string()
                : raw_images.string();

        // Current DatasetLoader can derive frame timestamps from timestamp-named images.
        config.frame_timestamps_path.clear();
    }

    vio::CameraIntrinsics intrinsics;
    intrinsics.fx = 356.3485;
    intrinsics.fy = 418.1912;
    intrinsics.cx = 363.0043;
    intrinsics.cy = 250.2713;

    config.camera_intrinsics = intrinsics;

    config.output_poses_csv = "../results/poses.csv";
    config.output_observations_csv = "../results/observations.csv";
    config.output_video_path = options.video_out;
    config.output_landmarks_csv = "../results/landmarks.csv";

    config.gravity = Eigen::Vector3d(0.0, 0.0, 9.81);
    config.tracker_win_size = 9;
    config.tracker_max_level = 3;
    config.tracker_max_iters = 10;
    config.tracker_eps = 1e-3f;
    config.stream_realtime = options.realtime;
    config.stream_rate = options.stream_rate;
    config.stream_max_image_queue = options.max_image_queue;

    PointsVideoLogger points_video_logger(config.output_video_path);

#if defined(VIO_ENABLE_RERUN_VISUALIZATION)
    vio::RerunVisualizer visualizer("vio_pipeline");
    if (options.rerun) {
        if (!visualizer.connect()) {
            std::cerr << "warning: Rerun receiver is not reachable on 127.0.0.1:9877\n";
        }
    }
    std::vector<vio::FrameState> trajectory;
#endif

    if (options.points_video
#if defined(VIO_ENABLE_RERUN_VISUALIZATION)
        || options.rerun
#endif
    ) {
        config.frame_logger =
            [&](const vio::TrackedFrame& frame,
                const std::vector<vio::Track>& tracks,
                const std::vector<vio::Landmark>& landmarks,
                vio::VioStatus status,
                bool pose_reliable,
                const std::filesystem::path& image_path) {
                if (options.points_video) {
                    points_video_logger.log(frame, tracks, landmarks, status, image_path);
                }

#if defined(VIO_ENABLE_RERUN_VISUALIZATION)
                if (options.rerun) {
                    if (pose_reliable) {
                        trajectory.push_back(frame.state);
                        visualizer.logPose(frame.state);
                        visualizer.logTrajectory(trajectory);
                    }
                    visualizer.logLandmarks(landmarks);
                    visualizer.logTrackedFeatures(frame.observations);
                    visualizer.logImagePath(image_path, frame.state.timestamp);
                    visualizer.logStatus(status);
                }
#endif
            };
    }

    const vio::VioRunResult result = vio::VioPipeline::runConfigured(config);
    if (!result.success) {
        std::cerr << "VIO pipeline failed";
        if (!result.error.empty()) {
            std::cerr << ": " << result.error;
        }
        std::cerr << "\n";
        return 1;
    }

    std::cout << "VIO pipeline finished: "
              << result.frame_count << " frames, "
              << result.landmark_count << " landmarks\n";
    if (options.points_video) {
        std::cout << "Points video: " << config.output_video_path << "\n";
    }
    return 0;
}
