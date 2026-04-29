#include "frontend/pivot_frame.hpp"

#include <stdexcept>

namespace vio {

void PivotFrame::set(
    int frame_id,
    double timestamp,
    const cv::Mat& gray,
    const FrameState& pose,
    const std::vector<Track>& tracks
) {
    if (gray.empty()) {
        throw std::runtime_error("PivotFrame::set: empty image");
    }

    if (gray.type() != CV_8UC1) {
        throw std::runtime_error("PivotFrame::set: expected CV_8UC1 grayscale image");
    }

    frame_id_ = frame_id;
    timestamp_ = timestamp;
    gray_ = gray.clone();
    pose_ = pose;
    tracks_ = tracks;
    valid_ = true;
}

bool PivotFrame::isValid() const
{
    return valid_;
}

int PivotFrame::frameId() const
{
    return frame_id_;
}

double PivotFrame::timestamp() const
{
    return timestamp_;
}

const cv::Mat& PivotFrame::gray() const
{
    return gray_;
}

const FrameState& PivotFrame::pose() const
{
    return pose_;
}

const std::vector<Track>& PivotFrame::tracks() const
{
    return tracks_;
}

} // namespace vio
