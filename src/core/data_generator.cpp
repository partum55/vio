#include "core/data_generator.h"

#include <Eigen/Geometry>

#include <cmath>
#include <random>

namespace vio {

Trajectory generateTrajectory(const GeneratorConfig& config) {
    Trajectory traj;
    traj.reserve(config.num_poses);

    for (int i = 0; i < config.num_poses; ++i) {
        double t = static_cast<double>(i) / (config.num_poses - 1);
        double angle = 2.0 * M_PI * config.helix_turns * t;

        double x = config.helix_radius * std::cos(angle);
        double y = config.helix_radius * std::sin(angle);
        double z = config.helix_height * t;

        Eigen::Vector3d pos(x, y, z);
        Eigen::Vector3d target(0.0, 0.0, config.helix_height * 0.5);
        Eigen::Vector3d up(0.0, 0.0, 1.0);

        Eigen::Vector3d forward = (target - pos).normalized();
        Eigen::Vector3d right = forward.cross(up).normalized();
        Eigen::Vector3d cam_up = right.cross(forward).normalized();

        Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
        T_wc.block<3, 1>(0, 0) = right;
        T_wc.block<3, 1>(0, 1) = cam_up;
        T_wc.block<3, 1>(0, 2) = -forward;
        T_wc.block<3, 1>(0, 3) = pos;

        CameraPose pose;
        pose.timestamp = t;
        pose.T_wc = T_wc;
        traj.push_back(pose);
    }

    return traj;
}

PointCloud generatePointCloud(const GeneratorConfig& config) {
    PointCloud cloud;
    cloud.reserve(config.num_points);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_xz(-config.cloud_extent, config.cloud_extent);
    std::uniform_real_distribution<double> dist_y(0.0, config.cloud_height);

    for (int i = 0; i < config.num_points; ++i) {
        double x = dist_xz(rng);
        double y = dist_y(rng);
        double z = dist_xz(rng);

        float t = static_cast<float>(y / config.cloud_height);
        float r;
        float g;
        float b;
        if (t < 0.5f) {
            float s = t * 2.0f;
            r = 0.0f;
            g = s;
            b = 1.0f - s;
        } else {
            float s = (t - 0.5f) * 2.0f;
            r = s;
            g = 1.0f - s;
            b = 0.0f;
        }

        Point3D pt;
        pt.position = Eigen::Vector3d(x, y, z);
        pt.color = Eigen::Vector3f(r, g, b);
        cloud.push_back(pt);
    }

    return cloud;
}

} // namespace vio
