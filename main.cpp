#include "core/dataset.h"
#include "core/data_generator.h"
#include "core/rerun_stream.h"
#include "core/vio_runner.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

struct CliOptions {
    std::filesystem::path dataset_path;
    std::filesystem::path video_output = "first_view.mp4";
    std::string host = "127.0.0.1";
    int port = 9877;
    bool record_video = true;
    bool auto_spawn_receiver = true;
    bool synthetic_demo = false;
    bool vicon_demo = false;
    double playback_rate = 1.0;
};

void printUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " <dataset_path> [options]\n\n"
        << "Options:\n"
        << "  --help, -h               Show this help message\n"
        << "  --record <file>          Save first-view video to file (default: first_view.mp4)\n"
        << "  --no-record              Disable first-view video export\n"
        << "  --demo                   Run synthetic generated scene without a dataset\n"
        << "  --demo-l                 Alias for --vicon-demo (live Vicon position replay)\n"
        << "  --vicon-demo             Replay dataset/mav0/vicon0/data.csv as a live 3D point stream\n"
        << "  --rate <speed>           Playback speed multiplier for --vicon-demo (default: 1.0)\n"
        << "  --host <addr>            Receiver host (default: 127.0.0.1)\n"
        << "  --port <port>            Receiver port (default: 9877)\n"
        << "  --no-receiver            Do not auto-start the Python Rerun receiver\n";
}

CliOptions parseArgs(int argc, char** argv) {
    CliOptions options;
    bool dataset_consumed = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--record") {
            if (++i >= argc) {
                throw std::runtime_error("--record requires a file path");
            }
            options.video_output = argv[i];
            options.record_video = true;
            continue;
        }
        if (arg == "--no-record") {
            options.record_video = false;
            continue;
        }
        if (arg == "--demo") {
            options.synthetic_demo = true;
            continue;
        }
        if (arg == "--demo-l" || arg == "--vicon-demo" || arg == "--vicon-live-demo" || arg == "--vicon-live.demo") {
            options.vicon_demo = true;
            continue;
        }
        if (arg == "--rate") {
            if (++i >= argc) {
                throw std::runtime_error("--rate requires a value");
            }
            options.playback_rate = std::stod(argv[i]);
            continue;
        }
        if (arg == "--host") {
            if (++i >= argc) {
                throw std::runtime_error("--host requires a value");
            }
            options.host = argv[i];
            continue;
        }
        if (arg == "--port") {
            if (++i >= argc) {
                throw std::runtime_error("--port requires a value");
            }
            options.port = std::stoi(argv[i]);
            continue;
        }
        if (arg == "--no-receiver") {
            options.auto_spawn_receiver = false;
            continue;
        }
        if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown option: " + arg);
        }
        if (dataset_consumed) {
            throw std::runtime_error("only one dataset path is supported");
        }
        options.dataset_path = arg;
        dataset_consumed = true;
    }

    if (options.vicon_demo && (!dataset_consumed || options.dataset_path == ".")) {
        options.dataset_path = "dataset/mav0";
        dataset_consumed = true;
    }
    if (!dataset_consumed && !options.synthetic_demo) {
        throw std::runtime_error("dataset_path is required");
    }
    if (options.synthetic_demo && options.vicon_demo) {
        throw std::runtime_error("--demo and --vicon-demo cannot be used together");
    }
    if (options.playback_rate <= 0.0) {
        throw std::runtime_error("--rate must be positive");
    }
    return options;
}

std::filesystem::path executablePath() {
    std::vector<char> buffer(4096, '\0');
    const ssize_t len = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0) {
        return std::filesystem::current_path();
    }
    buffer[static_cast<std::size_t>(len)] = '\0';
    return std::filesystem::path(buffer.data());
}

pid_t spawnReceiver(const std::filesystem::path& repo_root,
                    const std::filesystem::path& log_dir,
                    const std::string& host,
                    int port) {
    const pid_t pid = ::fork();
    if (pid != 0) {
        return pid;
    }

    const std::filesystem::path python_dir = repo_root / "python";
    const std::string python_path = python_dir.string();
    ::setenv("PYTHONPATH", python_path.c_str(), 1);
    ::setenv("PYTHONUNBUFFERED", "1", 1);
    ::setenv("RUST_LOG", "off", 1);

    std::filesystem::create_directories(log_dir);
    const std::filesystem::path receiver_log = log_dir / "receiver.log";
    const int log_fd = ::open(
        receiver_log.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC,
        0644);
    if (log_fd >= 0) {
        ::dup2(log_fd, STDOUT_FILENO);
        ::dup2(log_fd, STDERR_FILENO);
        ::close(log_fd);
    }

    const std::string port_str = std::to_string(port);
    ::execlp("python3",
             "python3",
             "-m",
             "vio_py.cli",
             "--host",
             host.c_str(),
             "--port",
             port_str.c_str(),
             "--keep-alive",
             static_cast<char*>(nullptr));
    std::perror("execlp");
    _exit(127);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parseArgs(argc, argv);
        const std::filesystem::path repo_root = executablePath().parent_path().parent_path();
        const char* log_dir_env = std::getenv("VIO_LOG_DIR");
        const std::filesystem::path log_dir = log_dir_env && std::strlen(log_dir_env) > 0
            ? std::filesystem::path(log_dir_env)
            : (repo_root / ".run_logs");

        pid_t receiver_pid = -1;
        if (options.auto_spawn_receiver) {
            std::cout << "Starting Python Rerun receiver on "
                      << options.host << ":" << options.port << std::endl;
            std::cout << "Receiver log: " << (log_dir / "receiver.log") << std::endl;
            receiver_pid = spawnReceiver(repo_root, log_dir, options.host, options.port);
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        }

        vio::RerunStreamClient stream_client;
        bool stream_ready = stream_client.connect(options.host, options.port);
        if (!stream_ready) {
            std::cerr << "Warning: could not connect to Rerun receiver at "
                      << options.host << ":" << options.port
                      << ". Processing will continue without live visualization.\n";
        }

        vio::RunConfig run_config;
        run_config.host = options.host;
        run_config.port = options.port;
        run_config.stream_to_rerun = stream_ready;
        run_config.video_output = options.video_output;
        run_config.write_video = options.record_video;

        vio::RunResult result;
        if (options.synthetic_demo) {
            std::cout << "Running synthetic demo scene" << std::endl;
            vio::GeneratorConfig generator_config;
            result = vio::runSyntheticDemo(
                generator_config,
                run_config,
                stream_ready ? &stream_client : nullptr);
        } else if (options.vicon_demo) {
            std::cout << "Running live Vicon replay from " << options.dataset_path / "vicon0" / "data.csv"
                      << " at " << options.playback_rate << "x" << std::endl;
            if (options.record_video) {
                std::cout << "Video export is disabled for Vicon live replay." << std::endl;
            }

            run_config.write_video = false;
            vio::ViconReplayConfig replay_config;
            replay_config.playback_rate = options.playback_rate;
            result = vio::runViconLiveDemo(
                options.dataset_path,
                replay_config,
                run_config,
                stream_ready ? &stream_client : nullptr);
        } else {
            std::cout << "Loading dataset from " << options.dataset_path << std::endl;
            const vio::Dataset dataset = vio::loadEurocDataset(options.dataset_path);
            result = vio::runVisualInertialOdometry(
                dataset,
                run_config,
                stream_ready ? &stream_client : nullptr);
        }

        std::cout << "Processed frames: " << result.processed_frames << "\n";
        std::cout << "Streamed frames: " << result.streamed_frames << "\n";
        if (result.video_path.empty()) {
            std::cout << "Video export: disabled\n";
        } else {
            std::cout << "Video saved to: " << result.video_path << "\n";
        }

        if (stream_ready) {
            std::cout << "Rerun scene stays available in the spawned receiver process.\n";
        }
        if (receiver_pid > 0) {
            std::cout << "Receiver PID: " << receiver_pid << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }
}
