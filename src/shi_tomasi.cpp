#include "shi_tomasi.hpp"
#include "gaussian_blur.hpp"
#include "sobel.hpp"
#include "parallel_utils.hpp"

#include <algorithm>
#include <cmath>

using namespace std;

CustomShiTomasiDetector::CustomShiTomasiDetector(
    ABCThreadPool& pool,
    int num_tasks)
    : pool_(pool),
      num_tasks_(std::max(1, num_tasks))
{
}

int CustomShiTomasiDetector::effectiveTaskCount(const cv::Size& size) const
{
    const int pixels = size.width * size.height;

    if (pixels < 512 * 512)
    {
        return 1;
    }

    if (pixels < 1024 * 1024)
    {
        return std::min(num_tasks_, 4);
    }

    return std::max(1, num_tasks_);
}

cv::Mat CustomShiTomasiDetector::shiTomasiScoreImage(
    const cv::Mat& gray8,
    const ShiTomasiParams& p,
    int effective_tasks)
{
    CV_Assert(gray8.type() == CV_8U);

    gray8.convertTo(gray32_, CV_32F);

    blur_ = gray32_;
    const int blurPasses = std::max(1, static_cast<int>(std::round(p.gaussianSigma)));
    for (int i = 0; i < blurPasses; ++i)
    {
        gaussianBlurCustom(blur_, blur_tmp_, pool_, effective_tasks);
        std::swap(blur_, blur_tmp_);
    }

    centralDifferenceXY(blur_, Ix_, Iy_, pool_, effective_tasks);

    Ixx_.create(blur_.size(), CV_32F);
    Iyy_.create(blur_.size(), CV_32F);
    Ixy_.create(blur_.size(), CV_32F);

    parallel_for_rows(pool_, blur_.rows, effective_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                const float* ix = Ix_.ptr<float>(y);
                const float* iy = Iy_.ptr<float>(y);

                float* ixx = Ixx_.ptr<float>(y);
                float* iyy = Iyy_.ptr<float>(y);
                float* ixy = Ixy_.ptr<float>(y);

                for (int x = 0; x < blur_.cols; ++x)
                {
                    const float gx = ix[x];
                    const float gy = iy[x];
                    ixx[x] = gx * gx;
                    iyy[x] = gy * gy;
                    ixy[x] = gx * gy;
                }
            }
        });

    const int tensorPasses = std::max(1, p.blockSize / 2);
    for (int i = 0; i < tensorPasses; ++i)
    {
        gaussianBlurCustom(Ixx_, tensor_tmp_, pool_, effective_tasks);
        std::swap(Ixx_, tensor_tmp_);

        gaussianBlurCustom(Iyy_, tensor_tmp_, pool_, effective_tasks);
        std::swap(Iyy_, tensor_tmp_);

        gaussianBlurCustom(Ixy_, tensor_tmp_, pool_, effective_tasks);
        std::swap(Ixy_, tensor_tmp_);
    }

    score_.create(blur_.size(), CV_32F);

    parallel_for_rows(pool_, blur_.rows, effective_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                const float* ixx = Ixx_.ptr<float>(y);
                const float* iyy = Iyy_.ptr<float>(y);
                const float* ixy = Ixy_.ptr<float>(y);
                float* dst = score_.ptr<float>(y);

                for (int x = 0; x < blur_.cols; ++x)
                {
                    const float trace = ixx[x] + iyy[x];
                    const float det = ixx[x] * iyy[x] - ixy[x] * ixy[x];
                    const float halfTrace = 0.5f * trace;
                    const float inside = std::max(0.0f, halfTrace * halfTrace - det);
                    const float score = halfTrace - std::sqrt(inside);
                    dst[x] = std::max(0.0f, score);
                }
            }
        });

    return score_;
}

cv::Mat CustomShiTomasiDetector::nmsLocalMax(
    const cv::Mat& score,
    int r,
    int effective_tasks)
{
    CV_Assert(score.type() == CV_32F);

    if (r <= 0)
    {
        return score.clone();
    }

    cv::Mat out(score.size(), CV_32F, cv::Scalar(0));

    parallel_for_rows(pool_, score.rows, effective_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                float* dst = out.ptr<float>(y);

                for (int x = 0; x < score.cols; ++x)
                {
                    const float center = score.at<float>(y, x);
                    if (center <= 0.0f)
                    {
                        dst[x] = 0.0f;
                        continue;
                    }

                    bool isMax = true;

                    for (int yy = std::max(0, y - r);
                         yy <= std::min(score.rows - 1, y + r) && isMax;
                         ++yy)
                    {
                        const float* row = score.ptr<float>(yy);

                        for (int xx = std::max(0, x - r);
                             xx <= std::min(score.cols - 1, x + r);
                             ++xx)
                        {
                            if (yy == y && xx == x)
                            {
                                continue;
                            }

                            if (row[xx] > center)
                            {
                                isMax = false;
                                break;
                            }
                        }
                    }

                    dst[x] = isMax ? center : 0.0f;
                }
            }
        });

    return out;
}

std::vector<cv::Point2f> CustomShiTomasiDetector::selectWithGrid(
    const cv::Mat& scoreNms,
    const ShiTomasiParams& p)
{
    return selectWithGrid(scoreNms, p, cv::Mat());
}

std::vector<cv::Point2f> CustomShiTomasiDetector::selectWithGrid(
    const cv::Mat& scoreNms,
    const ShiTomasiParams& p,
    const cv::Mat& allowedMask)
{
    CV_Assert(allowedMask.empty() ||
              (allowedMask.type() == CV_8U && allowedMask.size() == scoreNms.size()));

    double minVal = 0.0;
    double maxVal = 0.0;

    if (allowedMask.empty())
    {
        cv::minMaxLoc(scoreNms, &minVal, &maxVal);
    }
    else
    {
        cv::minMaxLoc(scoreNms, &minVal, &maxVal, nullptr, nullptr, allowedMask);
    }

    if (maxVal <= 0.0)
    {
        return {};
    }

    const float thresh = static_cast<float>(p.qualityLevel * maxVal);

    std::vector<Candidate> cand;
    cand.reserve(scoreNms.rows * scoreNms.cols / 40);

    for (int y = 0; y < scoreNms.rows; ++y)
    {
        const float* row = scoreNms.ptr<float>(y);
        const uchar* maskRow = allowedMask.empty() ? nullptr : allowedMask.ptr<uchar>(y);

        for (int x = 0; x < scoreNms.cols; ++x)
        {
            if (maskRow && maskRow[x] == 0)
            {
                continue;
            }

            const float s = row[x];
            if (s >= thresh)
            {
                cand.push_back({s, x, y});
            }
        }
    }

    if (cand.empty())
    {
        return {};
    }

    const size_t keepCount = std::min<size_t>(
        cand.size(),
        std::max<size_t>(static_cast<size_t>(p.maxCorners) * 4,
                         static_cast<size_t>(p.maxCorners)));

    auto byScoreDesc = [](const Candidate& a, const Candidate& b)
    {
        return a.score > b.score;
    };

    if (cand.size() > keepCount)
    {
        std::nth_element(cand.begin(), cand.begin() + keepCount, cand.end(), byScoreDesc);
        cand.resize(keepCount);
    }

    std::sort(cand.begin(), cand.end(), byScoreDesc);

    std::vector<cv::Point2f> pts;
    pts.reserve(std::min<int>(p.maxCorners, static_cast<int>(cand.size())));

    const float minDist = static_cast<float>(p.minDistance);
    const float minDist2 = minDist * minDist;

    const int cellSize = std::max(1, static_cast<int>(std::ceil(minDist)));
    const int gridCols = (scoreNms.cols + cellSize - 1) / cellSize;
    const int gridRows = (scoreNms.rows + cellSize - 1) / cellSize;

    std::vector<std::vector<int>> grid(gridRows * gridCols);

    auto gridIndex = [gridCols](int gx, int gy)
    {
        return gy * gridCols + gx;
    };

    for (const auto& c : cand)
    {
        if (static_cast<int>(pts.size()) >= p.maxCorners)
        {
            break;
        }

        cv::Point2f pt(
            static_cast<float>(c.x),
            static_cast<float>(c.y));

        const int gx = c.x / cellSize;
        const int gy = c.y / cellSize;

        bool ok = true;

        for (int nny = std::max(0, gy - 1); nny <= std::min(gridRows - 1, gy + 1); ++nny)
        {
            for (int nnx = std::max(0, gx - 1); nnx <= std::min(gridCols - 1, gx + 1); ++nnx)
            {
                const auto& bucket = grid[gridIndex(nnx, nny)];

                for (int idx : bucket)
                {
                    const cv::Point2f& chosen = pts[idx];
                    const float dx = pt.x - chosen.x;
                    const float dy = pt.y - chosen.y;

                    if (dx * dx + dy * dy < minDist2)
                    {
                        ok = false;
                        break;
                    }
                }

                if (!ok)
                {
                    break;
                }
            }

            if (!ok)
            {
                break;
            }
        }

        if (ok)
        {
            const int newIndex = static_cast<int>(pts.size());
            pts.push_back(pt);
            grid[gridIndex(gx, gy)].push_back(newIndex);
        }
    }

    return pts;
}

std::vector<cv::Point2f> CustomShiTomasiDetector::detect(
    const cv::Mat& img,
    const ShiTomasiParams& params)
{
    return detect(img, params, cv::Mat());
}

std::vector<cv::Point2f> CustomShiTomasiDetector::detect(
    const cv::Mat& img,
    const ShiTomasiParams& params,
    const cv::Mat& allowedMask)
{
    cv::Mat gray = toGrayU8(img);
    return detectGray(gray, params, allowedMask);
}

std::vector<cv::Point2f> CustomShiTomasiDetector::detectGray(
    const cv::Mat& gray,
    const ShiTomasiParams& params)
{
    return detectGray(gray, params, cv::Mat());
}

std::vector<cv::Point2f> CustomShiTomasiDetector::detectGray(
    const cv::Mat& gray,
    const ShiTomasiParams& params,
    const cv::Mat& allowedMask)
{
    CV_Assert(gray.type() == CV_8U);
    CV_Assert(allowedMask.empty() ||
              (allowedMask.type() == CV_8U && allowedMask.size() == gray.size()));

    const int effective_tasks = effectiveTaskCount(gray.size());

    cv::Mat score = shiTomasiScoreImage(gray, params, effective_tasks);
    cv::Mat scoreNms = nmsLocalMax(score, params.nmsRadius, effective_tasks);

    return selectWithGrid(scoreNms, params, allowedMask);
}

std::vector<cv::Point2f> OpenCVShiTomasiDetector::detect(
    const cv::Mat& img,
    const ShiTomasiParams& params)
{
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

cv::Mat toGrayU8(const cv::Mat& imgBgrOrGray)
{
    cv::Mat gray;

    if (imgBgrOrGray.channels() == 3)
    {
        cv::cvtColor(imgBgrOrGray, gray, cv::COLOR_BGR2GRAY);
    }
    else if (imgBgrOrGray.channels() == 4)
    {
        cv::cvtColor(imgBgrOrGray, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        gray = imgBgrOrGray;
    }

    if (gray.type() != CV_8U)
    {
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
    int thickness)
{
    cv::Mat vis;

    if (imgBgrOrGray.channels() == 1)
    {
        cv::cvtColor(imgBgrOrGray, vis, cv::COLOR_GRAY2BGR);
    }
    else
    {
        vis = imgBgrOrGray.clone();
    }

    for (const auto& pt : pts)
    {
        cv::circle(
            vis,
            pt,
            radius,
            cv::Scalar(0, 255, 0),
            thickness,
            cv::LINE_AA);
    }

    return vis;
}
