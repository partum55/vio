#include "vio/data_generator.h"
#include "vio/data_loader.h"
#include "vio/rerun_stream.h"

#include <algorithm>
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
#include <sys/types.h>
#include <unistd.h>

namespace {

struct CliOptions {
    std::string trajectory_path;
    std::string cloud_path;
    std::string host = "127.0.0.1";
    int port = 9877;
    bool auto_spawn_receiver = true;
    double playback_speed = 1.0;
};

void printUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --help, -h              Show this help message\n"
        << "  --trajectory <file>     Load trajectory from TUM format file\n"
        << "  --cloud <file>          Load point cloud from XYZ text file\n"
        << "  --host <addr>           Receiver host (default: 127.0.0.1)\n"
        << "  --port <port>           Receiver port (default: 9877)\n"
        << "  --speed <value>         Playback speed multiplier (default: 1.0)\n"
        << "  --no-receiver           Do not auto-start the Python Rerun receiver\n"
        << "\n"
        << "If no --trajectory or --cloud is given, a synthetic demo is shown.\n"
        << "\n"
        << "File formats:\n"
        << "  Trajectory (TUM):  timestamp tx ty tz qx qy qz qw\n"
        << "  Point cloud (XYZ): x y z [r g b]   (RGB 0-255, optional)\n"
        << "  Lines starting with '#' are treated as comments.\n"
        << "\n"
        << "Examples:\n"
        << "  " << prog << " --cloud points.xyz\n"
        << "  " << prog << " --trajectory traj.tum --cloud points.xyz\n"
        << "  " << prog << " --trajectory traj.tum --speed 2.0\n";
}

CliOptions parseArgs(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--trajectory") {
            if (++i >= argc) {
                throw std::runtime_error("--trajectory requires a file path");
            }
            options.trajectory_path = argv[i];
            continue;
        }
        if (arg == "--cloud") {
            if (++i >= argc) {
                throw std::runtime_error("--cloud requires a file path");
            }
            options.cloud_path = argv[i];
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
        if (arg == "--speed") {
            if (++i >= argc) {
                throw std::runtime_error("--speed requires a value");
            }
            options.playback_speed = std::stod(argv[i]);
            continue;
        }
        if (arg == "--no-receiver") {
            options.auto_spawn_receiver = false;
            continue;
        }
        throw std::runtime_error("unknown option: " + arg);
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

std::size_t streamTrajectory(vio::RerunStreamClient& client,
                             const vio::Trajectory& trajectory,
                             double playback_speed) {
    if (trajectory.empty()) {
        client.sendDone(0);
        return 0;
    }

    std::size_t streamed = 0;
    for (std::size_t i = 0; i < trajectory.size(); ++i) {
        const vio::CameraPose& pose = trajectory[i];
        vio::StreamSample sample;
        sample.timestamp_s = pose.timestamp;
        sample.position = pose.T_wc.block<3, 1>(0, 3);
        sample.orientation = Eigen::Quaterniond(pose.T_wc.block<3, 3>(0, 0)).normalized();
        sample.track_count = 0;
        if (!client.sendSample(sample)) {
            break;
        }
        ++streamed;

        if (i + 1 < trajectory.size()) {
            double dt = trajectory[i + 1].timestamp - pose.timestamp;
            dt = std::max(0.0, dt);
            const double speed = playback_speed > 1e-6 ? playback_speed : 1.0;
            std::this_thread::sleep_for(std::chrono::duration<double>(dt / speed));
        }
    }
    client.sendDone(streamed);
    return streamed;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parseArgs(argc, argv);
        const bool use_synthetic = options.trajectory_path.empty() && options.cloud_path.empty();

        vio::Trajectory trajectory;
        vio::PointCloud cloud;

        if (use_synthetic) {
            std::cout << "No data files specified - running synthetic demo.\n";
            vio::GeneratorConfig generator_config;
            trajectory = vio::generateTrajectory(generator_config);
            cloud = vio::generatePointCloud(generator_config);
        } else {
            if (!options.trajectory_path.empty()) {
                trajectory = vio::loadTrajectoryTUM(options.trajectory_path);
            }
            if (!options.cloud_path.empty()) {
                cloud = vio::loadPointCloudXYZ(options.cloud_path);
            }
        }

        const std::filesystem::path repo_root = executablePath().parent_path().parent_path();
        pid_t receiver_pid = -1;
        if (options.auto_spawn_receiver) {
            std::cout << "Starting Python Rerun receiver on "
                      << options.host << ":" << options.port << "\n";
            receiver_pid = spawnReceiver(repo_root, options.host, options.port);
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        }

        vio::RerunStreamClient stream_client;
        if (!stream_client.connect(options.host, options.port)) {
            throw std::runtime_error(
                "could not connect to Rerun receiver at " + options.host + ":" + std::to_string(options.port));
        }

        const std::string scene_name = use_synthetic ? "synthetic_demo" : "trajectory_cloud";
        stream_client.sendInit(scene_name);
        if (!cloud.empty()) {
            stream_client.sendPointCloud(cloud);
        }

        const std::size_t streamed_frames = streamTrajectory(
            stream_client,
            trajectory,
            options.playback_speed);

        std::cout << "Streamed frames: " << streamed_frames << "\n";
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
