#include "keypoints/gaussian_blur.hpp"
#include "keypoints/shi_tomasi.hpp"
#include "keypoints/sobel.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace vio {

CustomShiTomasiDetector::CustomShiTomasiDetector(
    ABCThreadPool& pool,
    int num_tasks)
    : pool_(pool),
      num_tasks_(std::max(1, num_tasks)) {}

cv::Mat CustomShiTomasiDetector::shiTomasiScoreImage(
    const cv::Mat& gray8,
    const ShiTomasiParams& p) {
    CV_Assert(gray8.type() == CV_8U);

    cv::Mat gray32;
    gray8.convertTo(gray32, CV_32F);

    blur_ = gray32.clone();

    const int blurPasses = std::max(1, static_cast<int>(std::round(p.gaussianSigma)));
    for (int i = 0; i < blurPasses; ++i) {
        blur_ = gaussianBlurCustom(blur_, pool_, num_tasks_);
    }

    centralDifferenceXY(blur_, Ix_, Iy_, pool_, num_tasks_);

    cv::multiply(Ix_, Ix_, Ixx_);
    cv::multiply(Iy_, Iy_, Iyy_);
    cv::multiply(Ix_, Iy_, Ixy_);

    const int tensorPasses = std::max(1, p.blockSize / 2);
    for (int i = 0; i < tensorPasses; ++i) {
        Ixx_ = gaussianBlurCustom(Ixx_, pool_, num_tasks_);
        Iyy_ = gaussianBlurCustom(Iyy_, pool_, num_tasks_);
        Ixy_ = gaussianBlurCustom(Ixy_, pool_, num_tasks_);
    }

    trace_ = Ixx_ + Iyy_;
    det_ = Ixx_.mul(Iyy_) - Ixy_.mul(Ixy_);

    halfTrace_ = 0.5f * trace_;
    inside_ = halfTrace_.mul(halfTrace_) - det_;

    cv::max(inside_, 0, inside_);
    cv::sqrt(inside_, sqrtInside_);

    score_ = halfTrace_ - sqrtInside_;
    cv::max(score_, 0, score_);

    return score_;
}

cv::Mat CustomShiTomasiDetector::nmsLocalMax(const cv::Mat& score, int r) {
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2 * r + 1, 2 * r + 1));

    cv::dilate(score, dilated_, kernel);

    cv::Mat isMax = (score == dilated_);
    cv::Mat out = cv::Mat::zeros(score.size(), score.type());
    score.copyTo(out, isMax);
    return out;
}

std::vector<cv::Point2f> CustomShiTomasiDetector::selectWithGrid(
    const cv::Mat& scoreNms,
    const ShiTomasiParams& p) {
    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(scoreNms, &minVal, &maxVal);

    const float thresh = static_cast<float>(p.qualityLevel * maxVal);

    std::vector<Candidate> cand;
    cand.reserve(scoreNms.rows * scoreNms.cols / 20);

    for (int y = 0; y < scoreNms.rows; ++y) {
        const float* row = scoreNms.ptr<float>(y);
        for (int x = 0; x < scoreNms.cols; ++x) {
            const float s = row[x];
            if (s >= thresh) {
                cand.push_back({s, x, y});
            }
        }
    }

    std::sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });

    std::vector<cv::Point2f> pts;
    pts.reserve(std::min<int>(p.maxCorners, static_cast<int>(cand.size())));

    const float minDist = static_cast<float>(p.minDistance);
    const float minDist2 = minDist * minDist;

    const int cellSize = std::max(1, static_cast<int>(std::ceil(minDist)));
    const int gridCols = (scoreNms.cols + cellSize - 1) / cellSize;
    const int gridRows = (scoreNms.rows + cellSize - 1) / cellSize;

    std::vector<std::vector<int>> grid(gridRows * gridCols);

    auto gridIndex = [gridCols](int gx, int gy) {
        return gy * gridCols + gx;
    };

    for (const auto& c : cand) {
        if (static_cast<int>(pts.size()) >= p.maxCorners) {
            break;
        }

        cv::Point2f pt(static_cast<float>(c.x), static_cast<float>(c.y));

        int gx = c.x / cellSize;
        int gy = c.y / cellSize;
        bool ok = true;

        for (int nny = std::max(0, gy - 1); nny <= std::min(gridRows - 1, gy + 1); ++nny) {
            for (int nnx = std::max(0, gx - 1); nnx <= std::min(gridCols - 1, gx + 1); ++nnx) {
                const auto& bucket = grid[gridIndex(nnx, nny)];
                for (int idx : bucket) {
                    const cv::Point2f& chosen = pts[idx];
                    const float dx = pt.x - chosen.x;
                    const float dy = pt.y - chosen.y;

                    if (dx * dx + dy * dy < minDist2) {
                        ok = false;
                        break;
                    }
                }

                if (!ok) {
                    break;
                }
            }

            if (!ok) {
                break;
            }
        }

        if (ok) {
            int newIndex = static_cast<int>(pts.size());
            pts.push_back(pt);
            grid[gridIndex(gx, gy)].push_back(newIndex);
        }
    }

    return pts;
}

std::vector<cv::Point2f> CustomShiTomasiDetector::detect(
    const cv::Mat& img,
    const ShiTomasiParams& params) {
    cv::Mat gray = toGrayU8(img);
    cv::Mat score = shiTomasiScoreImage(gray, params);
    cv::Mat scoreNms = nmsLocalMax(score, params.nmsRadius);

    return selectWithGrid(scoreNms, params);
}

std::vector<cv::Point2f> OpenCVShiTomasiDetector::detect(
    const cv::Mat& img,
    const ShiTomasiParams& params) {
    cv::Mat gray = toGrayU8(img);

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(
        gray,
        pts,
        params.maxCorners,
        params.qualityLevel,
        params.minDistance,
        cv::noArray(),
        params.blockSize,
        false,
        0.04);

    return pts;
}

cv::Mat toGrayU8(const cv::Mat& imgBgrOrGray) {
    cv::Mat gray;

    if (imgBgrOrGray.channels() == 3) {
        cv::cvtColor(imgBgrOrGray, gray, cv::COLOR_BGR2GRAY);
    } else if (imgBgrOrGray.channels() == 4) {
        cv::cvtColor(imgBgrOrGray, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = imgBgrOrGray;
    }

    if (gray.type() != CV_8U) {
        cv::Mat tmp;
        gray.convertTo(tmp, CV_8U);
        return tmp;
    }

    return gray;
}

cv::Mat drawKeypointsOnImage(
    const cv::Mat& imgBgrOrGray,
    const std::vector<cv::Point2f>& pts,
    int radius,
    int thickness) {
    cv::Mat vis;

    if (imgBgrOrGray.channels() == 1) {
        cv::cvtColor(imgBgrOrGray, vis, cv::COLOR_GRAY2BGR);
    } else {
        vis = imgBgrOrGray.clone();
    }

    for (const auto& p : pts) {
        cv::circle(
            vis,
            p,
            radius,
            cv::Scalar(0, 255, 0),
            thickness,
            cv::LINE_AA);
    }

    return vis;
}

} // namespace vio
