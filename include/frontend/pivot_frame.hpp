#pragma once

#include "core/types.hpp"
#include "tracking/tracking_vis.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

class PivotFrame {
public:
    void set(
        int frame_id,
        double timestamp,
        const cv::Mat& gray,
        const vio::FrameState& pose,
        const std::vector<Track>& tracks
    );

    bool isValid() const;

    int frameId() const;
    double timestamp() const;

    const cv::Mat& gray() const;
    const vio::FrameState& pose() const;
    const std::vector<Track>& tracks() const;

private:
    bool valid_ = false;

    int frame_id_ = -1;
    double timestamp_ = 0.0;

    cv::Mat gray_;
    vio::FrameState pose_;
    std::vector<Track> tracks_;
};