#include "imu.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

Eigen::Quaterniond deltaQuat(const Eigen::Vector3d& omega, double dt)
{
    double angle = omega.norm() * dt; //omega - швидкість
    if (angle < 1e-12) {
        return Eigen::Quaterniond(1,0,0,0);
    }

    const Eigen::Vector3d axis = omega.normalized();
    const Eigen::AngleAxisd aa(angle, axis); // обертання на angle навколо axis
    Eigen::Quaterniond dq(aa);
    dq.normalize();
    return dq;
}

class KalmanAxis {
public:
    Eigen::Vector2d mu; // [value, derivative]
    Eigen::Matrix2d Sigma;

    double sigma_meas;
    double q_process;

    KalmanAxis(double sigma_meas_ = 0.01, double q_process_ = 1.0)
        : sigma_meas(sigma_meas_), q_process(q_process_)
    {
        mu.setZero();
        Sigma.setIdentity();
    }

    void init(double first_measurement) {
        mu << first_measurement, 0.0;
        Sigma.setIdentity();
    }

    void predict(double dt) {
        Eigen::Matrix2d A;
        A << 1.0, dt,
             0.0, 1.0;

        Eigen::Matrix2d Q; //шум процесу 
        Q << dt * dt * dt / 3.0, dt * dt / 2.0,
             dt * dt / 2.0, dt;

        Q *= q_process;

        mu = A * mu;
        Sigma = A * Sigma * A.transpose() + Q;
    }

    double update(double z_meas) {
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

void integrateImuRaw(const std::vector<ImuSample>& imu, double t0, double t1, Pose& pose, const Eigen::Vector3d& gravity, std::vector<Pose>& trajectory_out)
{
    trajectory_out.clear();
    trajectory_out.reserve(10000);
    
    size_t idx = 0;
    while (idx < imu.size() && imu[idx].t < t0) idx++;
    if (idx == imu.size()) return;

    pose.t = imu[idx].t;
    while (idx < imu.size() && imu[idx].t <= t1) {
        const auto& s = imu[idx];
        double dt = s.t - pose.t;
        if (dt <= 0) { idx++; continue; }

        pose.q *= deltaQuat(s.gyro, dt);
        pose.q.normalize();

        Eigen::Vector3d acc_world = pose.q * s.acc;
        Eigen::Vector3d acc_lin = acc_world - gravity;

        pose.v += acc_lin * dt;
        pose.p += pose.v * dt;

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
        std::cerr << "Could not open IMU CSV: " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        double t_ns, gx, gy, gz, ax, ay, az;
        char comma;

        ss >> t_ns >> comma >> gx   >> comma >> gy   >> comma >> gz   >> comma >> ax   >> comma >> ay   >> comma >> az;

        if (ss.fail()) continue;

        ImuSample s;
        s.t = t_ns * 1e-9;
        s.gyro = {gx, gy, gz};
        s.acc  = {ax, ay, az};

        out.push_back(s);
    }

    return !out.empty();
}

bool saveTrajectoryCsv(const std::string& path, const std::vector<Pose>& traj)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;

    out << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz\n";
    out << std::fixed << std::setprecision(9);

    for (const auto& p : traj) {
        out << p.t << "," << p.p.x() << "," << p.p.y() << "," << p.p.z() << "," << p.v.x() << "," << p.v.y() << "," << p.v.z() << "," << p.q.w() << "," << p.q.x() << "," << p.q.y() << "," << p.q.z() << "\n";
    }
    return true;
}
