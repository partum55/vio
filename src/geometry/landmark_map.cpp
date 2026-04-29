#include "geometry/landmark_map.hpp"

#include <algorithm>

namespace vio {

void LandmarkMap::clear() {
    landmarks_by_track_.clear();
}

bool LandmarkMap::empty() const {
    return landmarks_by_track_.empty();
}

std::size_t LandmarkMap::size() const {
    return landmarks_by_track_.size();
}

void LandmarkMap::addOrUpdate(
    int track_id,
    const Eigen::Vector3d& p_w,
    double reprojection_error,
    int num_observations
) {
    if (track_id < 0 || !p_w.allFinite()) {
        return;
    }

    auto it = landmarks_by_track_.find(track_id);

    if (it == landmarks_by_track_.end()) {
        Landmark landmark;
        landmark.id = static_cast<int>(landmarks_by_track_.size());
        landmark.track_id = track_id;
        landmark.p_w = p_w;
        landmark.valid = true;
        landmark.reprojection_error = reprojection_error;
        landmark.num_observations = num_observations;

        landmarks_by_track_[track_id] = landmark;
        return;
    }

    Landmark& landmark = it->second;
    landmark.p_w = p_w;
    landmark.valid = true;
    landmark.reprojection_error = reprojection_error;
    landmark.num_observations = std::max(landmark.num_observations, num_observations);
}

bool LandmarkMap::hasTrack(int track_id) const {
    return landmarks_by_track_.find(track_id) != landmarks_by_track_.end();
}

const Landmark* LandmarkMap::getByTrackId(int track_id) const {
    auto it = landmarks_by_track_.find(track_id);
    if (it == landmarks_by_track_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<Landmark> LandmarkMap::getValidLandmarks() const {
    std::vector<Landmark> result;
    result.reserve(landmarks_by_track_.size());

    for (const auto& item : landmarks_by_track_) {
        const Landmark& landmark = item.second;
        if (landmark.valid) {
            result.push_back(landmark);
        }
    }

    return result;
}

void LandmarkMap::removeInvalid() {
    for (auto it = landmarks_by_track_.begin(); it != landmarks_by_track_.end();) {
        if (!it->second.valid) {
            it = landmarks_by_track_.erase(it);
        } else {
            ++it;
        }
    }
}

void LandmarkMap::buildPnPCorrespondences(
    const std::vector<Observation>& observations,
    std::vector<Eigen::Vector3d>& points_3d_w,
    std::vector<Eigen::Vector2d>& points_2d
) const {
    points_3d_w.clear();
    points_2d.clear();

    for (const Observation& obs : observations) {
        if (!obs.valid) {
            continue;
        }

        const Landmark* landmark = getByTrackId(obs.track_id);
        if (landmark == nullptr || !landmark->valid) {
            continue;
        }

        points_3d_w.push_back(landmark->p_w);
        points_2d.push_back(obs.uv);
    }
}

} // namespace vio
