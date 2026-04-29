#include "tracking/tracking_vis.hpp"

namespace vio {

float pointDistance(const cv::Point2f& a, const cv::Point2f& b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool isFarFromExisting(
    const cv::Point2f& p,
    const std::vector<Track>& tracks,
    const float minDist
) {
    for (const auto& t : tracks) {
        if (pointDistance(p, t.pt) < minDist) {
            return false;
        }
    }
    return true;
}

void addNewTracks(
    const cv::Mat& gray,
    std::vector<Track>& tracks,
    int& nextTrackId,
    const int maxCorners,
    const double qualityLevel,
    const double minDistance
) {
    std::vector<cv::Point2f> detected;
    cv::goodFeaturesToTrack(gray, detected, maxCorners, qualityLevel, minDistance);

    for (const auto& p : detected) {
        if (isFarFromExisting(p, tracks, static_cast<float>(minDistance))) {
            Track t;
            t.id = nextTrackId++;
            t.pt = p;
            t.history.push_back(p);
            tracks.push_back(t);
        }
    }
}

cv::Mat drawTrackingVisualization(
    const cv::Mat& frame,
    const std::vector<Track>& tracks,
    const int tailLength
) {
    cv::Mat vis = frame.clone();

    for (const auto& t : tracks) {
        const int histSize = static_cast<int>(t.history.size());
        const int startIdx = std::max(0, histSize - tailLength);

        for (int i = startIdx + 1; i < histSize; ++i) {
            cv::line(vis, t.history[i - 1], t.history[i], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        cv::circle(vis, t.pt, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
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
