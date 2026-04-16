#include "triangulation/triangulator.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <unordered_map>
#include <vector>
#include <limits>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace
{
    constexpr int kMinFrameGap = 2;
    constexpr int kMaxFrameGap = 12;
    constexpr int kLocalSupportPadding = 8;
    constexpr double kPreferredBaseline = 0.20;
    constexpr double kMaxReasonableBaseline = 1.50;

    struct TrackObservation
    {
        const vio::TrackedFrame* frame = nullptr;
        const vio::Observation* obs = nullptr;
    };

    Eigen::Matrix<double, 3, 4> buildProjectionMatrix(
        const vio::FrameState& state,
        const CameraIntrinsics& intrinsics
    )
    {
        const Eigen::Matrix3d K = intrinsics.matrix();

        const Eigen::Matrix3d R_wc = state.q_wc.toRotationMatrix();
        const Eigen::Matrix3d R_cw = R_wc.transpose();
        const Eigen::Vector3d t_cw = -R_cw * state.t_wc;

        Eigen::Matrix<double, 3, 4> P;
        P.leftCols<3>() = K * R_cw;
        P.rightCols<1>() = K * t_cw;

        return P;
    }

    bool triangulateTwoViewLinear(
        const Eigen::Matrix<double, 3, 4>& P1,
        const Eigen::Matrix<double, 3, 4>& P2,
        const Eigen::Vector2d& uv1,
        const Eigen::Vector2d& uv2,
        Eigen::Vector3d& X_w
    )
    {
        Eigen::Matrix4d A;
        A.row(0) = uv1.x() * P1.row(2) - P1.row(0);
        A.row(1) = uv1.y() * P1.row(2) - P1.row(1);
        A.row(2) = uv2.x() * P2.row(2) - P2.row(0);
        A.row(3) = uv2.y() * P2.row(2) - P2.row(1);

        const Eigen::JacobiSVD<Eigen::Matrix4d> svd(
            A,
            Eigen::ComputeFullV
        );

        const Eigen::Vector4d X_h = svd.matrixV().col(3);

        if (std::abs(X_h(3)) < 1e-12) {
            return false;
        }

        X_w = X_h.head<3>() / X_h(3);
        return X_w.allFinite();
    }

    bool projectPointToImage(
        const vio::FrameState& state,
        const CameraIntrinsics& intrinsics,
        const Eigen::Vector3d& X_w,
        Eigen::Vector2d& uv,
        double& depth
    )
    {
        const Eigen::Matrix3d R_wc = state.q_wc.toRotationMatrix();
        const Eigen::Matrix3d R_cw = R_wc.transpose();
        const Eigen::Vector3d t_cw = -R_cw * state.t_wc;

        const Eigen::Vector3d X_c = R_cw * X_w + t_cw;
        depth = X_c.z();

        if (depth <= 0.0) {
            return false;
        }

        const double x = X_c.x() / depth;
        const double y = X_c.y() / depth;

        uv.x() = intrinsics.fx * x + intrinsics.cx;
        uv.y() = intrinsics.fy * y + intrinsics.cy;

        return std::isfinite(uv.x()) && std::isfinite(uv.y());
    }

    bool hasPositiveDepth(
        const vio::FrameState& state,
        const Eigen::Vector3d& X_w,
        double min_depth
    )
    {
        const Eigen::Matrix3d R_wc = state.q_wc.toRotationMatrix();
        const Eigen::Matrix3d R_cw = R_wc.transpose();
        const Eigen::Vector3d t_cw = -R_cw * state.t_wc;

        const Eigen::Vector3d X_c = R_cw * X_w + t_cw;
        return X_c.z() > min_depth;
    }

    double computeMeanReprojectionError(
        const std::vector<TrackObservation>& track_observations,
        const CameraIntrinsics& intrinsics,
        const Eigen::Vector3d& X_w,
        double min_depth,
        bool& all_positive_depth
    )
    {
        double error_sum = 0.0;
        int count = 0;
        all_positive_depth = true;

        for (const auto& item : track_observations) {
            Eigen::Vector2d uv_proj;
            double depth = 0.0;

            const bool ok = projectPointToImage(
                item.frame->state,
                intrinsics,
                X_w,
                uv_proj,
                depth
            );

            if (!ok || depth <= min_depth) {
                all_positive_depth = false;
                return std::numeric_limits<double>::infinity();
            }

            const Eigen::Vector2d diff = uv_proj - item.obs->uv;
            error_sum += diff.norm();
            ++count;
        }

        if (count == 0) {
            all_positive_depth = false;
            return std::numeric_limits<double>::infinity();
        }

        return error_sum / static_cast<double>(count);
    }

    bool selectPairForTriangulation(
        const std::vector<TrackObservation>& track_observations,
        double min_baseline,
        int& best_i,
        int& best_j,
        double& best_baseline
    )
    {
        best_i = -1;
        best_j = -1;
        best_baseline = -1.0;

        double best_score = std::numeric_limits<double>::infinity();

        for (int i = 0; i < static_cast<int>(track_observations.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(track_observations.size()); ++j) {
                const int frame_gap =
                    std::abs(track_observations[j].frame->state.frame_id -
                             track_observations[i].frame->state.frame_id);

                if (frame_gap < kMinFrameGap || frame_gap > kMaxFrameGap) {
                    continue;
                }

                const Eigen::Vector3d& c1 = track_observations[i].frame->state.t_wc;
                const Eigen::Vector3d& c2 = track_observations[j].frame->state.t_wc;
                const double baseline = (c1 - c2).norm();

                if (baseline < min_baseline) {
                    continue;
                }

                double score = std::abs(baseline - kPreferredBaseline);

                if (baseline > kMaxReasonableBaseline) {
                    score += 1000.0 + baseline;
                }

                if (score < best_score) {
                    best_score = score;
                    best_i = i;
                    best_j = j;
                    best_baseline = baseline;
                }
            }
        }

        if (best_i >= 0 && best_j >= 0) {
            return true;
        }

        double fallback_best_baseline = std::numeric_limits<double>::infinity();

        for (int i = 0; i < static_cast<int>(track_observations.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(track_observations.size()); ++j) {
                const Eigen::Vector3d& c1 = track_observations[i].frame->state.t_wc;
                const Eigen::Vector3d& c2 = track_observations[j].frame->state.t_wc;
                const double baseline = (c1 - c2).norm();

                if (baseline < min_baseline) {
                    continue;
                }

                if (baseline < fallback_best_baseline) {
                    fallback_best_baseline = baseline;
                    best_i = i;
                    best_j = j;
                    best_baseline = baseline;
                }
            }
        }

        return best_i >= 0 && best_j >= 0;
    }

    std::vector<TrackObservation> buildLocalSupport(
        const std::vector<TrackObservation>& track_observations,
        int best_i,
        int best_j
    )
    {
        std::vector<TrackObservation> local_support;
        local_support.reserve(track_observations.size());

        const int id_lo = std::min(
            track_observations[best_i].frame->state.frame_id,
            track_observations[best_j].frame->state.frame_id
        ) - kLocalSupportPadding;

        const int id_hi = std::max(
            track_observations[best_i].frame->state.frame_id,
            track_observations[best_j].frame->state.frame_id
        ) + kLocalSupportPadding;

        for (const auto& item : track_observations) {
            const int fid = item.frame->state.frame_id;
            if (fid >= id_lo && fid <= id_hi) {
                local_support.push_back(item);
            }
        }

        if (local_support.size() < 2) {
            local_support.push_back(track_observations[best_i]);
            local_support.push_back(track_observations[best_j]);
        }

        return local_support;
    }
}

std::vector<vio::Landmark> triangulateLandmarks(
    const std::vector<vio::TrackedFrame>& sequence,
    const CameraIntrinsics& camera_intrinsics,
    const TriangulationParams& params
)
{
    std::vector<vio::Landmark> landmarks;

    if (sequence.empty() || !camera_intrinsics.isValid()) {
        std::cout << "\n=== TRIANGULATION DEBUG ===\n";
        std::cout << "Sequence empty or intrinsics invalid\n";
        std::cout << "===========================\n\n";
        return landmarks;
    }

    std::unordered_map<int, std::vector<TrackObservation>> observations_by_track;

    for (const auto& frame : sequence) {
        for (const auto& obs : frame.observations) {
            if (!obs.valid) {
                continue;
            }

            observations_by_track[obs.track_id].push_back(
                TrackObservation{&frame, &obs}
            );
        }
    }

    int rejected_min_obs = 0;
    int rejected_pair = 0;
    int rejected_baseline = 0;
    int rejected_linear = 0;
    int rejected_depth1 = 0;
    int rejected_depth2 = 0;
    int rejected_reproj_depth = 0;
    int rejected_reproj_err = 0;
    int accepted = 0;

    double min_track_len = std::numeric_limits<double>::infinity();
    double max_track_len = 0.0;
    double sum_track_len = 0.0;

    double min_best_baseline = std::numeric_limits<double>::infinity();
    double max_best_baseline = 0.0;
    double sum_best_baseline = 0.0;
    int baseline_count = 0;

    landmarks.reserve(observations_by_track.size());

    for (const auto& [track_id, track_observations] : observations_by_track) {
        const int obs_count = static_cast<int>(track_observations.size());

        min_track_len = std::min(min_track_len, static_cast<double>(obs_count));
        max_track_len = std::max(max_track_len, static_cast<double>(obs_count));
        sum_track_len += static_cast<double>(obs_count);

        if (obs_count < params.min_observations) {
            ++rejected_min_obs;
            continue;
        }

        int best_i = -1;
        int best_j = -1;
        double best_baseline = -1.0;

        if (!selectPairForTriangulation(
                track_observations,
                params.min_baseline,
                best_i,
                best_j,
                best_baseline)) {
            ++rejected_pair;
            continue;
        }

        min_best_baseline = std::min(min_best_baseline, best_baseline);
        max_best_baseline = std::max(max_best_baseline, best_baseline);
        sum_best_baseline += best_baseline;
        ++baseline_count;

        if (best_baseline < params.min_baseline) {
            ++rejected_baseline;
            continue;
        }

        const auto& obs1 = track_observations[best_i];
        const auto& obs2 = track_observations[best_j];

        const Eigen::Matrix<double, 3, 4> P1 =
            buildProjectionMatrix(obs1.frame->state, camera_intrinsics);
        const Eigen::Matrix<double, 3, 4> P2 =
            buildProjectionMatrix(obs2.frame->state, camera_intrinsics);

        Eigen::Vector3d X_w;
        if (!triangulateTwoViewLinear(P1, P2, obs1.obs->uv, obs2.obs->uv, X_w)) {
            ++rejected_linear;
            continue;
        }

        if (!hasPositiveDepth(obs1.frame->state, X_w, params.min_depth)) {
            ++rejected_depth1;
            continue;
        }

        if (!hasPositiveDepth(obs2.frame->state, X_w, params.min_depth)) {
            ++rejected_depth2;
            continue;
        }

        const std::vector<TrackObservation> local_support =
            buildLocalSupport(track_observations, best_i, best_j);

        bool all_positive_depth = true;
        const double mean_reprojection_error = computeMeanReprojectionError(
            local_support,
            camera_intrinsics,
            X_w,
            params.min_depth,
            all_positive_depth
        );

        if (!all_positive_depth) {
            ++rejected_reproj_depth;
            continue;
        }

        if (!std::isfinite(mean_reprojection_error) ||
            mean_reprojection_error > params.max_reprojection_error) {
            ++rejected_reproj_err;
            continue;
        }

        vio::Landmark landmark;
        landmark.track_id = track_id;
        landmark.p_w = X_w;
        landmark.valid = true;
        landmark.reprojection_error = mean_reprojection_error;
        landmark.num_observations = static_cast<int>(local_support.size());

        landmarks.push_back(landmark);
        ++accepted;
    }

    if (!observations_by_track.empty()) {
        std::cout << "track len min/avg/max: "
                  << min_track_len << " / "
                  << (sum_track_len / static_cast<double>(observations_by_track.size())) << " / "
                  << max_track_len << "\n";
    } else {
        std::cout << "track len min/avg/max: 0 / 0 / 0\n";
    }

    if (baseline_count > 0) {
        std::cout << "selected baseline min/avg/max: "
                  << min_best_baseline << " / "
                  << (sum_best_baseline / static_cast<double>(baseline_count)) << " / "
                  << max_best_baseline << "\n";
    } else {
        std::cout << "selected baseline min/avg/max: 0 / 0 / 0\n";
    }

    return landmarks;
}