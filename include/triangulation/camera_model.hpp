#pragma once

#include <Eigen/Dense>

struct CameraIntrinsics
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    Eigen::Matrix3d matrix() const
    {
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(0, 2) = cx;
        K(1, 2) = cy;
        return K;
    }

    bool isValid() const
    {
        return fx > 0.0 && fy > 0.0;
    }
};