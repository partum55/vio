#include "frontend/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vio {

FeatureExtractor::FeatureExtractor()
    : backend_(VisionComputeBackend::createAuto())
{
}

std::vector<cv::Point2f> FeatureExtractor::extract(
    const cv::Mat& gray,
    const ShiTomasiParams& params
) {
    if (gray.empty()) {
        throw std::runtime_error("FeatureExtractor::extract: empty image");
    }

    if (gray.type() != CV_8UC1) {
        throw std::runtime_error("FeatureExtractor::extract: expected CV_8UC1 grayscale image");
    }

    return extract(gray, params, cv::Mat{});
}

std::vector<cv::Point2f> FeatureExtractor::extract(
    const cv::Mat& gray,
    const ShiTomasiParams& params,
    const cv::Mat& mask
) {
    if (gray.empty()) {
        throw std::runtime_error("FeatureExtractor::extract: empty image");
    }

    if (gray.type() != CV_8UC1) {
        throw std::runtime_error("FeatureExtractor::extract: expected CV_8UC1 grayscale image");
    }

    if (!mask.empty() && (mask.type() != CV_8UC1 || mask.size() != gray.size())) {
        throw std::runtime_error("FeatureExtractor::extract: invalid mask");
    }

    const cv::Mat score = scoreImage(gray, params);
    return selectFeatures(score, params, mask);
}

cv::Mat FeatureExtractor::scoreImage(
    const cv::Mat& gray,
    const ShiTomasiParams& params
) const {
    cv::Mat gray32;
    gray.convertTo(gray32, CV_32F);

    cv::Mat blur = gray32;
    cv::Mat tmp;
    const int blur_passes = std::max(1, static_cast<int>(std::round(params.gaussianSigma)));
    for (int i = 0; i < blur_passes; ++i) {
        backend_->gaussianBlur(blur, tmp);
        blur = tmp;
    }

    cv::Mat ix;
    cv::Mat iy;
    backend_->sobelGradients(blur, ix, iy);

    cv::Mat ixx(blur.size(), CV_32F);
    cv::Mat iyy(blur.size(), CV_32F);
    cv::Mat ixy(blur.size(), CV_32F);

    for (int y = 0; y < blur.rows; ++y) {
        const float* ix_row = ix.ptr<float>(y);
        const float* iy_row = iy.ptr<float>(y);
        float* ixx_row = ixx.ptr<float>(y);
        float* iyy_row = iyy.ptr<float>(y);
        float* ixy_row = ixy.ptr<float>(y);

        for (int x = 0; x < blur.cols; ++x) {
            const float gx = ix_row[x];
            const float gy = iy_row[x];
            ixx_row[x] = gx * gx;
            iyy_row[x] = gy * gy;
            ixy_row[x] = gx * gy;
        }
    }

    const int tensor_passes = std::max(1, params.blockSize / 2);
    for (int i = 0; i < tensor_passes; ++i) {
        backend_->gaussianBlur(ixx, tmp);
        ixx = tmp;
        backend_->gaussianBlur(iyy, tmp);
        iyy = tmp;
        backend_->gaussianBlur(ixy, tmp);
        ixy = tmp;
    }

    cv::Mat score(blur.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < blur.rows; ++y) {
        const float* ixx_row = ixx.ptr<float>(y);
        const float* iyy_row = iyy.ptr<float>(y);
        const float* ixy_row = ixy.ptr<float>(y);
        float* score_row = score.ptr<float>(y);

        for (int x = 0; x < blur.cols; ++x) {
            const float trace = ixx_row[x] + iyy_row[x];
            const float det = ixx_row[x] * iyy_row[x] - ixy_row[x] * ixy_row[x];
            const float half_trace = 0.5f * trace;
            const float inside = std::max(0.0f, half_trace * half_trace - det);
            score_row[x] = std::max(0.0f, half_trace - std::sqrt(inside));
        }
    }

    return score;
}

std::vector<cv::Point2f> FeatureExtractor::selectFeatures(
    const cv::Mat& score,
    const ShiTomasiParams& params,
    const cv::Mat& mask
) const {
    double max_value = 0.0;
    if (mask.empty()) {
        cv::minMaxLoc(score, nullptr, &max_value);
    } else {
        cv::minMaxLoc(score, nullptr, &max_value, nullptr, nullptr, mask);
    }

    if (max_value <= 0.0) {
        return {};
    }

    const float threshold = static_cast<float>(params.qualityLevel * max_value);

    struct Candidate {
        float score = 0.0f;
        int x = 0;
        int y = 0;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(score.rows * score.cols / 40));

    const int r = std::max(0, params.nmsRadius);
    for (int y = 0; y < score.rows; ++y) {
        const float* score_row = score.ptr<float>(y);
        const uchar* mask_row = mask.empty() ? nullptr : mask.ptr<uchar>(y);

        for (int x = 0; x < score.cols; ++x) {
            if (mask_row != nullptr && mask_row[x] == 0) {
                continue;
            }

            const float value = score_row[x];
            if (value < threshold) {
                continue;
            }

            bool is_max = true;
            for (int yy = std::max(0, y - r); yy <= std::min(score.rows - 1, y + r) && is_max; ++yy) {
                const float* row = score.ptr<float>(yy);
                for (int xx = std::max(0, x - r); xx <= std::min(score.cols - 1, x + r); ++xx) {
                    if ((yy != y || xx != x) && row[xx] > value) {
                        is_max = false;
                        break;
                    }
                }
            }

            if (is_max) {
                candidates.push_back(Candidate{value, x, y});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });

    std::vector<cv::Point2f> points;
    points.reserve(static_cast<std::size_t>(std::max(0, params.maxCorners)));
    const float min_distance = static_cast<float>(params.minDistance);
    const float min_distance2 = min_distance * min_distance;

    for (const Candidate& candidate : candidates) {
        if (static_cast<int>(points.size()) >= params.maxCorners) {
            break;
        }

        cv::Point2f point(
            static_cast<float>(candidate.x),
            static_cast<float>(candidate.y)
        );

        bool far_enough = true;
        for (const cv::Point2f& existing : points) {
            const cv::Point2f diff = point - existing;
            if (diff.dot(diff) < min_distance2) {
                far_enough = false;
                break;
            }
        }

        if (far_enough) {
            points.push_back(point);
        }
    }

    return points;
}

} // namespace vio
