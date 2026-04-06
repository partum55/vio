#include "imu.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

Eigen::Quaterniond deltaQuat(const Eigen::Vector3d& omega, double dt)
{
    double angle = omega.norm() * dt;
    if (angle < 1e-12) {
        return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    }

    Eigen::Vector3d axis = omega.normalized();
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

Eigen::Quaterniond initialOrientationFromAccel(const Eigen::Vector3d& acc_meas, const Eigen::Vector3d& gravity_world)
{
    if (acc_meas.norm() < 1e-12 || gravity_world.norm() < 1e-12) {
        return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    }

    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(acc_meas.normalized(), gravity_world.normalized());
    q.normalize();
    return q;
}

class Kalman {
public:
    Eigen::Vector2d mu;
    Eigen::Matrix2d Sigma;

    double sigma_meas;
    double q_process;

    Kalman(double sigma_meas_ = 0.01, double q_process_ = 1.0)
        : sigma_meas(sigma_meas_), q_process(q_process_)
    {
        mu << 0.0, 0.0;
        Sigma << 1.0, 0.0, 0.0, 1.0;
    }

    void init(double first_measurement)
    {
        mu << first_measurement, 0.0;
        Sigma << sigma_meas * sigma_meas, 0.0, 0.0, 1.0;
    }

    void predict(double dt)
    {
        Eigen::Matrix2d A;
        A << 1.0, dt,
             0.0, 1.0;

        Eigen::Matrix2d Q;
        Q << dt * dt * dt / 3.0, dt * dt / 2.0, dt * dt / 2.0, dt;

        Q *= q_process;

        mu = A * mu;
        Sigma = A * Sigma * A.transpose() + Q;
    }

    double update(double z_meas)
    {
        Eigen::RowVector2d H;
        H << 1.0, 0.0;

        double R = sigma_meas * sigma_meas;
        double innovation = z_meas - (H * mu)(0);
        double S = (H * Sigma * H.transpose())(0, 0) + R;
        Eigen::Vector2d K = Sigma * H.transpose() / S;

        mu = mu + K * innovation;
        Sigma = (Eigen::Matrix2d::Identity() - K * H) * Sigma;

        return mu(0);
    }
};

static size_t findStartIndex(const std::vector<ImuSample>& imu, double t0)
{
    size_t idx = 0;
    while (idx < imu.size() && imu[idx].t < t0) {
        idx++;
    }
    return idx;
}

void integrateImuFiltered(const std::vector<ImuSample>& imu, double t0, double t1, Pose& pose, const Eigen::Vector3d& gravity, std::vector<Pose>& trajectory_out)
{
    trajectory_out.clear();
    trajectory_out.reserve(imu.size());

    size_t idx = findStartIndex(imu, t0);
    if (idx >= imu.size()) {
        return;
    }

    pose.t = imu[idx].t;
    pose.p.setZero();
    pose.v.setZero();
    pose.q = initialOrientationFromAccel(imu[idx].acc, gravity);
    pose.q.normalize();

    const double gyro_noise_density  = 1.6968e-04;
    const double gyro_random_walk    = 1.9393e-05;
    const double accel_noise_density = 2.0e-03;
    const double accel_random_walk   = 3.0e-03;


    const double rate_hz = 200.0;
    const double gyro_q_scale  = 10.0;
    const double accel_q_scale = 10.0;

    const double gyro_sigma  = gyro_noise_density  * std::sqrt(rate_hz);
    const double accel_sigma = accel_noise_density * std::sqrt(rate_hz);

    const double gyro_q  = gyro_q_scale  * gyro_random_walk  * gyro_random_walk;
    const double accel_q = accel_q_scale * accel_random_walk * accel_random_walk;

    Kalman kf_gx(gyro_sigma,  gyro_q);
    Kalman kf_gy(gyro_sigma,  gyro_q);
    Kalman kf_gz(gyro_sigma,  gyro_q);

    Kalman kf_ax(accel_sigma, accel_q);
    Kalman kf_ay(accel_sigma, accel_q);
    Kalman kf_az(accel_sigma, accel_q);

    kf_gx.init(imu[idx].gyro.x());
    kf_gy.init(imu[idx].gyro.y());
    kf_gz.init(imu[idx].gyro.z());

    kf_ax.init(imu[idx].acc.x());
    kf_ay.init(imu[idx].acc.y());
    kf_az.init(imu[idx].acc.z());

    while (idx < imu.size() && imu[idx].t <= t1) {
        const ImuSample& s = imu[idx];
        double dt = s.t - pose.t;

        if (dt <= 0.0) {
            idx++;
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

        Eigen::Vector3d acc_world = pose.q * acc_filtered;
        Eigen::Vector3d acc_lin = acc_world - gravity;

        Eigen::Vector3d v_prev = pose.v;
        pose.v += acc_lin * dt;
        pose.p += v_prev * dt + 0.5 * acc_lin * dt * dt;

        pose.t = s.t;
        trajectory_out.push_back(pose);
        idx++;
    }
}

void integrateImuRaw(const std::vector<ImuSample>& imu, double t0, double t1, Pose& pose, const Eigen::Vector3d& gravity, std::vector<Pose>& trajectory_out)
{
    trajectory_out.clear();
    trajectory_out.reserve(imu.size());

    size_t idx = findStartIndex(imu, t0);
    if (idx >= imu.size()) {
        return;
    }

    pose.t = imu[idx].t;
    pose.p.setZero();
    pose.v.setZero();
    pose.q = initialOrientationFromAccel(imu[idx].acc, gravity);
    pose.q.normalize();

    while (idx < imu.size() && imu[idx].t <= t1) {
        const ImuSample& s = imu[idx];
        double dt = s.t - pose.t;
        if (dt <= 0.0) {
            idx++;
            continue;
        }

        pose.q *= deltaQuat(s.gyro, dt);
        pose.q.normalize();

        Eigen::Vector3d acc_world = pose.q * s.acc;
        Eigen::Vector3d acc_lin = acc_world - gravity;

        Eigen::Vector3d v_prev = pose.v;
        pose.v += acc_lin * dt;
        pose.p += v_prev * dt + 0.5 * acc_lin * dt * dt;

        pose.t = s.t;
        trajectory_out.push_back(pose);
        idx++;
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
        s.gyro = Eigen::Vector3d(gx, gy, gz);
        s.acc = Eigen::Vector3d(ax, ay, az);
        out.push_back(s);
    }

    return !out.empty();
}

bool saveTrajectoryCsv(const std::string& path, const std::vector<Pose>& traj)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    out << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz\n";
    out << std::fixed << std::setprecision(9);

    for (const Pose& p : traj) {
        out << p.t << ","
            << p.p.x() << "," << p.p.y() << "," << p.p.z() << ","
            << p.v.x() << "," << p.v.y() << "," << p.v.z() << ","
            << p.q.w() << "," << p.q.x() << "," << p.q.y() << "," << p.q.z() << "\n";
    }

    return true;
}
