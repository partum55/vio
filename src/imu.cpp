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

void integrateImu(const std::vector<ImuSample>& imu, size_t& idx, double t0, double t1, Pose& pose, const Eigen::Vector3d& gravity_world)
{
    double t = t0;

    while (idx < imu.size() && imu[idx].t <= t1) {
        const auto& s = imu[idx];

        if (s.t <= t) { idx++; continue; }

        const double dt = s.t - t;

        pose.q *= deltaQuat(s.gyro, dt);
        pose.q.normalize();

        Eigen::Vector3d acc_world = pose.q * s.acc;
        Eigen::Vector3d acc_lin = acc_world - gravity_world;

        // euler
        pose.v += acc_lin * dt;
        pose.p += pose.v * dt;

        t = s.t;
        idx++;
    }
    pose.t = t1;
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