#pragma once

#include "core/tracked_frame.hpp"
#include "frontend/pivot_frame.hpp"
#include "geometry/geometry_backend.hpp"
#include "geometry/landmark_map.hpp"

#include <vector>

namespace vio {

    struct VioPipelineParams {
        int min_tracked_points = 20;
        int min_landmarks_for_pnp = 20;
        int min_landmarks_after_initial_triangulation = 6;

        // Visual-only fallback: use robust 2D parallax between the pivot and
        // current frame when the IMU pose prior is unavailable/weak.
        double min_pixel_baseline = 1.0;
        double max_pixel_baseline = 45.0;
        int min_shared_tracks_for_baseline = 15;

        // Reject obviously broken pose jumps before triangulation/PnP refresh.
        double max_pose_baseline = 1.2;

        // PnP must have enough RANSAC inliers to be allowed to update the pose.
        int min_pnp_inliers = 14;
    };

    class VioPipeline {
    public:
        explicit VioPipeline(
            GeometryBackend geometry,
            const VioPipelineParams& params = VioPipelineParams{}
        );

        void reset();
        void processFrame(const TrackedFrame& input_frame);

        const std::vector<TrackedFrame>& frames() const;
        const LandmarkMap& landmarks() const;

    private:
        enum class Stage {
            NeedPivot,
            NeedInitialLandmarks,
            TrackWithLandmarks
        };

        int validObservationCount(const TrackedFrame& frame) const;
        int pnpCorrespondenceCount(const TrackedFrame& frame) const;

        double poseBaseline(const TrackedFrame& current) const;
        double robustPixelBaseline(const TrackedFrame& current, int* shared_count = nullptr) const;
        bool baselineReady(const TrackedFrame& current) const;
        bool baselineReasonable(const TrackedFrame& current) const;

        void setPivot(const TrackedFrame& frame, Stage next_stage);
        void initializePivot(const TrackedFrame& frame);
        bool triangulateFromPivot(const TrackedFrame& current, int min_created);
        bool refinePoseWithPnP(TrackedFrame& current);
        void resetTrackingSegment(const TrackedFrame& current);

    private:
        GeometryBackend geometry_;
        VioPipelineParams params_;

        Stage stage_ = Stage::NeedPivot;
        PivotFrame pivot_;
        LandmarkMap landmark_map_;

        std::vector<TrackedFrame> frames_;
    };

} // namespace vio
