#include "visualization/points_video_renderer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <string>

namespace vio {

cv::Mat drawPointsVideoFrame(
    const cv::Mat& frame,
    const std::vector<Track>& tracks,
    const int tail_length
) {
    cv::Mat vis = frame.clone();

    for (const Track& track : tracks) {
        const int hist_size = static_cast<int>(track.history.size());
        const int start_idx = std::max(0, hist_size - tail_length);

        for (int i = start_idx + 1; i < hist_size; ++i) {
            cv::line(
                vis,
                track.history[i - 1],
                track.history[i],
                cv::Scalar(0, 255, 0),
                1,
                cv::LINE_AA
            );
        }

        cv::circle(vis, track.pt, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    cv::putText(
        vis,
        "Active tracks: " + std::to_string(tracks.size()),
        cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    return vis;
}

} // namespace vio
