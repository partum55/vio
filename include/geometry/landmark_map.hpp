#pragma once

#include "core/types.hpp"
#include "geometry/landmark.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace vio {

    class LandmarkMap {
    public:
        void clear();

        bool empty() const;
        std::size_t size() const;

        void addOrUpdate(
            int track_id,
            const Eigen::Vector3d& p_w,
            double reprojection_error = 0.0,
            int num_observations = 1
        );

        bool hasTrack(int track_id) const;

        const Landmark* getByTrackId(int track_id) const;

        std::vector<Landmark> getValidLandmarks() const;

        void removeInvalid();

        void buildPnPCorrespondences(
            const std::vector<Observation>& observations,
            std::vector<Eigen::Vector3d>& points_3d_w,
            std::vector<Eigen::Vector2d>& points_2d
        ) const;

    private:
        std::unordered_map<int, Landmark> landmarks_by_track_;
    };

}