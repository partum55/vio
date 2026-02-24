#include "vio/data_generator.h"
#include "vio/data_loader.h"
#include "vio/viewer.h"

#include <cstring>
#include <iostream>
#include <string>

static void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "\n"
        "Options:\n"
        "  --help, -h              Show this help message\n"
        "  --trajectory <file>     Load trajectory from TUM format file\n"
        "  --cloud <file>          Load point cloud from XYZ text file\n"
        "  --record <file>         Record to video file (default: output.mp4)\n"
        "  --width <pixels>        Video width  (default: 1920)\n"
        "  --height <pixels>       Video height (default: 1080)\n"
        "  --fps <rate>            Video FPS    (default: 60)\n"
        "\n"
        "If no --trajectory or --cloud is given, a synthetic demo is shown.\n"
        "\n"
        "File formats:\n"
        "  Trajectory (TUM):  timestamp tx ty tz qx qy qz qw\n"
        "  Point cloud (XYZ): x y z [r g b]   (RGB 0-255, optional)\n"
        "  Lines starting with '#' are treated as comments.\n"
        "\n"
        "Examples:\n"
        "  " << prog << "                                       # synthetic demo\n"
        "  " << prog << " --trajectory traj.tum                 # trajectory only\n"
        "  " << prog << " --cloud points.xyz                    # point cloud only\n"
        "  " << prog << " --trajectory t.tum --cloud c.xyz      # both\n"
        "  " << prog << " --record out.mp4 --trajectory t.tum   # record with data\n";
}

int main(int argc, char** argv) {
    std::string trajectory_path;
    std::string cloud_path;
    std::string record_path;
    int width = 1920;
    int height = 1080;
    int fps = 60;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--trajectory") {
            if (++i >= argc) { std::cerr << "Error: --trajectory requires a file path\n"; return 1; }
            trajectory_path = argv[i];
        } else if (arg == "--cloud") {
            if (++i >= argc) { std::cerr << "Error: --cloud requires a file path\n"; return 1; }
            cloud_path = argv[i];
        } else if (arg == "--record") {
            if (++i >= argc) { std::cerr << "Error: --record requires a file path\n"; return 1; }
            record_path = argv[i];
        } else if (arg == "--width") {
            if (++i >= argc) { std::cerr << "Error: --width requires a value\n"; return 1; }
            width = std::stoi(argv[i]);
        } else if (arg == "--height") {
            if (++i >= argc) { std::cerr << "Error: --height requires a value\n"; return 1; }
            height = std::stoi(argv[i]);
        } else if (arg == "--fps") {
            if (++i >= argc) { std::cerr << "Error: --fps requires a value\n"; return 1; }
            fps = std::stoi(argv[i]);
        } else {
            std::cerr << "Error: unknown option '" << arg << "'\n\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Load or generate data
    bool use_synthetic = trajectory_path.empty() && cloud_path.empty();

    vio::Trajectory trajectory;
    vio::PointCloud cloud;

    if (use_synthetic) {
        std::cout << "No data files specified — running synthetic demo.\n";
        vio::GeneratorConfig gen_cfg;
        trajectory = vio::generateTrajectory(gen_cfg);
        cloud = vio::generatePointCloud(gen_cfg);
    } else {
        if (!trajectory_path.empty())
            trajectory = vio::loadTrajectoryTUM(trajectory_path);
        if (!cloud_path.empty())
            cloud = vio::loadPointCloudXYZ(cloud_path);
    }

    vio::Viewer viewer(trajectory, cloud);

    if (!record_path.empty()) {
        vio::RecordConfig rec_cfg;
        rec_cfg.output_path = record_path;
        rec_cfg.width = width;
        rec_cfg.height = height;
        rec_cfg.fps = fps;
        std::cout << "Recording video to: " << rec_cfg.output_path << "\n";
        viewer.record(rec_cfg);
    } else {
        viewer.run();
    }

    return 0;
}
