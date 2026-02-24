#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct ImuSample {
    double t = 0.0;  //в секундах
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero(); //rad/s
    Eigen::Vector3d acc  = Eigen::Vector3d::Zero(); //з гравітацією
};

struct Pose {
    double t = 0.0;
    Eigen::Quaterniond q = Eigen::Quaterniond(1,0,0,0);
    Eigen::Vector3d v = Eigen::Vector3d::Zero(); 
    Eigen::Vector3d p = Eigen::Vector3d::Zero(); 
};


Eigen::Quaterniond deltaQuat(const Eigen::Vector3d& omega, double dt);


void integrateImu(const std::vector<ImuSample>& imu, size_t& idx, double t0, double t1, Pose& pose, const Eigen::Vector3d& gravity_world)

bool loadImuCsv(const std::string& path, std::vector<ImuSample>& out);
bool saveTrajectoryCsv(const std::string& path, const std::vector<Pose>& traj);