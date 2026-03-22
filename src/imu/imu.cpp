#include "imu/imu.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace vio {

Eigen::Quaterniond deltaQuat(const Eigen::Vector3d& omega, double dt)
{
    const double angle = omega.norm() * dt;
    if (angle < 1e-12) {
        return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    }

    const Eigen::Vector3d axis = omega.normalized();
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

class Kalman {
public:
    Eigen::Vector2d mu;
    Eigen::Matrix2d Sigma;

    double sigma_meas;
    double q_process;

    Kalman(double sigma_meas_ = 0.01, double q_process_ = 1.0)
        : mu(Eigen::Vector2d::Zero())
        , Sigma(Eigen::Matrix2d::Identity())
        , sigma_meas(sigma_meas_)
        , q_process(q_process_)
    {
    }

    void init(double first_measurement)
    {
        mu << first_measurement, 0.0;
        Sigma << sigma_meas * sigma_meas, 0.0,
                 0.0, 1.0;
    }

    void predict(double dt)
    {
        Eigen::Matrix2d A;
        A << 1.0, dt,
             0.0, 1.0;

        Eigen::Matrix2d Q;
        Q << dt * dt * dt / 3.0, dt * dt / 2.0,
             dt * dt / 2.0, dt;
        Q *= q_process;

        mu = A * mu;
        Sigma = A * Sigma * A.transpose() + Q;
    }

    double update(double z_meas)
    {
        Eigen::RowVector2d H;
        H << 1.0, 0.0;

        const double R = sigma_meas * sigma_meas;
        const double innovation = z_meas - (H * mu)(0);
        const double S = (H * Sigma * H.transpose())(0, 0) + R;

        const Eigen::Vector2d K = Sigma * H.transpose() / S;

        mu = mu + K * innovation;
        Sigma = (Eigen::Matrix2d::Identity() - K * H) * Sigma;

        return mu(0);
    }
};

void integrateImuFiltered(const std::vector<ImuSample>& imu,
                          double t0,
                          double t1,
                          ImuPose& pose,
                          const Eigen::Vector3d& gravity,
                          std::vector<ImuPose>& trajectory_out)
{
    trajectory_out.clear();
    trajectory_out.reserve(imu.size());

    std::size_t idx = 0;
    while (idx < imu.size() && imu[idx].t < t0) {
        ++idx;
    }

    if (idx == imu.size()) {
        return;
    }

    Kalman kf_gx(0.02, 1.2), kf_gy(0.02, 1.2), kf_gz(0.02, 1.2);
    Kalman kf_ax(0.20, 6.0), kf_ay(0.20, 6.0), kf_az(0.20, 6.0);

    const auto& first = imu[idx];
    kf_gx.init(first.gyro.x());
    kf_gy.init(first.gyro.y());
    kf_gz.init(first.gyro.z());

    kf_ax.init(first.acc.x());
    kf_ay.init(first.acc.y());
    kf_az.init(first.acc.z());

    pose.t = first.t;

    while (idx < imu.size() && imu[idx].t <= t1) {
        const auto& s = imu[idx];
        const double dt = s.t - pose.t;

        if (dt <= 0.0) {
            ++idx;
            continue;
        }

        kf_gx.predict(dt);
        kf_gy.predict(dt);
        kf_gz.predict(dt);
        kf_ax.predict(dt);
        kf_ay.predict(dt);
        kf_az.predict(dt);

        Eigen::Vector3d gyro_filtered;
        Eigen::Vector3d acc_filtered;

        gyro_filtered.x() = kf_gx.update(s.gyro.x());
        gyro_filtered.y() = kf_gy.update(s.gyro.y());
        gyro_filtered.z() = kf_gz.update(s.gyro.z());

        acc_filtered.x() = kf_ax.update(s.acc.x());
        acc_filtered.y() = kf_ay.update(s.acc.y());
        acc_filtered.z() = kf_az.update(s.acc.z());

        pose.q *= deltaQuat(gyro_filtered, dt);
        pose.q.normalize();

        const Eigen::Vector3d acc_world = pose.q * acc_filtered;
        const Eigen::Vector3d acc_lin = acc_world - gravity;

        const Eigen::Vector3d v_prev = pose.v;
        pose.v += acc_lin * dt;
        pose.p += v_prev * dt + 0.5 * acc_lin * dt * dt;

        pose.t = s.t;
        trajectory_out.push_back(pose);
        ++idx;
    }
}

void integrateImuRaw(const std::vector<ImuSample>& imu,
                     double t0,
                     double t1,
                     ImuPose& pose,
                     const Eigen::Vector3d& gravity,
                     std::vector<ImuPose>& trajectory_out)
{
    trajectory_out.clear();
    trajectory_out.reserve(imu.size());

    std::size_t idx = 0;
    while (idx < imu.size() && imu[idx].t < t0) {
        ++idx;
    }

    if (idx == imu.size()) {
        return;
    }

    pose.t = imu[idx].t;
    while (idx < imu.size() && imu[idx].t <= t1) {
        const auto& s = imu[idx];
        const double dt = s.t - pose.t;

        if (dt <= 0.0) {
            ++idx;
            continue;
        }

        pose.q *= deltaQuat(s.gyro, dt);
        pose.q.normalize();

        const Eigen::Vector3d acc_world = pose.q * s.acc;
        const Eigen::Vector3d acc_lin = acc_world - gravity;

        const Eigen::Vector3d v_prev = pose.v;
        pose.v += acc_lin * dt;
        pose.p += v_prev * dt + 0.5 * acc_lin * dt * dt;

        pose.t = s.t;
        trajectory_out.push_back(pose);
        ++idx;
    }
}

bool loadImuCsv(const std::string& path, std::vector<ImuSample>& out)
{
    out.clear();

    std::ifstream f(path);
    if (!f) {
        std::cerr << "Couldn't open IMU csv: " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        double t_ns, gx, gy, gz, ax, ay, az;
        char comma;

        ss >> t_ns >> comma >> gx >> comma >> gy >> comma >> gz >> comma >> ax >> comma >> ay >> comma >> az;
        if (ss.fail()) {
            continue;
        }

        ImuSample s;
        s.t = t_ns * 1e-9;
        s.gyro = {gx, gy, gz};
        s.acc = {ax, ay, az};
        out.push_back(s);
    }

    return !out.empty();
}

bool saveTrajectoryCsv(const std::string& path, const std::vector<ImuPose>& traj)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    out << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz\n";
    out << std::fixed << std::setprecision(9);

    for (const auto& p : traj) {
        out << p.t << ","
            << p.p.x() << "," << p.p.y() << "," << p.p.z() << ","
            << p.v.x() << "," << p.v.y() << "," << p.v.z() << ","
            << p.q.w() << "," << p.q.x() << "," << p.q.y() << "," << p.q.z()
            << "\n";
    }

    return true;
}

} // namespace vio
