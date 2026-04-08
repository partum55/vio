#include "shi_tomasi.hpp"
#include "feature_refresh.hpp"
#include "tpool_default.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    std::vector<cv::Point2f> takePrefix(
        const std::vector<cv::Point2f>& pts,
        std::size_t count)
    {
        const std::size_t n = std::min(count, pts.size());
        return std::vector<cv::Point2f>(pts.begin(), pts.begin() + n);
    }

    cv::Mat drawMaskOverlay(const cv::Mat& img, const cv::Mat& mask)
    {
        cv::Mat vis;
        if (img.channels() == 1)
        {
            cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
        }
        else
        {
            vis = img.clone();
        }

        for (int y = 0; y < mask.rows; ++y)
        {
            const uchar* maskRow = mask.ptr<uchar>(y);
            cv::Vec3b* visRow = vis.ptr<cv::Vec3b>(y);

            for (int x = 0; x < mask.cols; ++x)
            {
                if (maskRow[x] == 0)
                {
                    visRow[x][0] = static_cast<uchar>(visRow[x][0] * 0.5);
                    visRow[x][1] = static_cast<uchar>(visRow[x][1] * 0.5);
                    visRow[x][2] = static_cast<uchar>(
                        std::min(255.0, visRow[x][2] * 0.5 + 100.0));
                }
            }
        }

        return vis;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: compare_keypoints <image_path> [num_threads]\n";
        return 1;
    }

    std::filesystem::create_directories("shi_tomasi_outputs");

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Failed to read image: " << argv[1] << "\n";
        return 1;
    }

    int num_threads = 0;
    if (argc >= 3)
    {
        num_threads = std::stoi(argv[2]);
    }
    else
    {
        const unsigned int hw = std::thread::hardware_concurrency();
        num_threads = (hw == 0) ? 4 : static_cast<int>(hw);
    }

    if (num_threads <= 0)
    {
        std::cerr << "Invalid num_threads: " << num_threads << "\n";
        return 1;
    }

    ThreadPool pool(num_threads);
    ABCThreadPool& abstract_pool = pool;

    ShiTomasiParams p;
    p.maxCorners = 500;
    p.qualityLevel = 0.01;
    p.minDistance = 10.0;
    p.blockSize = 5;
    p.gaussianSigma = 1.0;
    p.nmsRadius = 2;

    CustomShiTomasiDetector myDetector(abstract_pool, num_threads);
    OpenCVShiTomasiDetector cvDetector;

    std::vector<cv::Point2f> pts_mine = myDetector.detect(img, p);
    std::vector<cv::Point2f> pts_cv = cvDetector.detect(img, p);

    cv::Mat vis_mine = drawKeypointsOnImage(img, pts_mine);
    cv::Mat vis_cv = drawKeypointsOnImage(img, pts_cv);

    cv::putText(
        vis_mine,
        "Mine",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA);

    cv::putText(
        vis_cv,
        "OpenCV",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA);

    const std::string out1 = "shi_tomasi_outputs/keypoints_mine.png";
    const std::string out2 = "shi_tomasi_outputs/keypoints_opencv.png";

    if (!cv::imwrite(out1, vis_mine))
    {
        std::cerr << "Failed to write " << out1 << "\n";
    }

    if (!cv::imwrite(out2, vis_cv))
    {
        std::cerr << "Failed to write " << out2 << "\n";
    }

    cv::Mat allowedMask(img.rows, img.cols, CV_8U, cv::Scalar(255));
    cv::rectangle(
        allowedMask,
        cv::Rect(0, 0, img.cols / 2, img.rows),
        cv::Scalar(0),
        -1);

    std::vector<cv::Point2f> pts_masked = myDetector.detect(img, p, allowedMask);

    cv::Mat vis_mask_overlay = drawMaskOverlay(img, allowedMask);
    cv::Mat vis_masked = drawKeypointsOnImage(vis_mask_overlay, pts_masked);

    cv::putText(
        vis_masked,
        "Masked detector test",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA);

    const std::string out3 = "shi_tomasi_outputs/keypoints_masked.png";
    const std::string out4 = "shi_tomasi_outputs/allowed_mask_overlay.png";

    if (!cv::imwrite(out3, vis_masked))
    {
        std::cerr << "Failed to write " << out3 << "\n";
    }

    if (!cv::imwrite(out4, vis_mask_overlay))
    {
        std::cerr << "Failed to write " << out4 << "\n";
    }

    int forbidden_count = 0;
    for (const auto& pt : pts_masked)
    {
        const int x = static_cast<int>(std::round(pt.x));
        const int y = static_cast<int>(std::round(pt.y));

        if (x >= 0 && x < allowedMask.cols &&
            y >= 0 && y < allowedMask.rows &&
            allowedMask.at<uchar>(y, x) == 0)
        {
            ++forbidden_count;
        }
    }

    std::vector<cv::Point2f> trackedSmall = takePrefix(pts_mine, 50);

    FeatureRefreshParams refreshParams;
    refreshParams.minTrackedFeatures = 200;
    refreshParams.targetFeatures = 300;
    refreshParams.suppressionRadius = 10.0f;

    std::vector<cv::Point2f> refreshedSmall = refreshFeaturesIfNeeded(
        img,
        trackedSmall,
        myDetector,
        p,
        refreshParams);

    cv::Mat vis_refresh_small = drawKeypointsOnImage(img, refreshedSmall);
    cv::putText(
        vis_refresh_small,
        "Refresh triggered (start=50)",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA);

    const std::string out5 = "shi_tomasi_outputs/refreshed_from_small_set.png";
    if (!cv::imwrite(out5, vis_refresh_small))
    {
        std::cerr << "Failed to write " << out5 << "\n";
    }

    std::vector<cv::Point2f> trackedLarge = takePrefix(pts_mine, 250);
    std::vector<cv::Point2f> refreshedLarge = refreshFeaturesIfNeeded(
        img,
        trackedLarge,
        myDetector,
        p,
        refreshParams);

    cv::Mat vis_refresh_large = drawKeypointsOnImage(img, refreshedLarge);
    cv::putText(
        vis_refresh_large,
        "Refresh skipped (start=250)",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA);

    const std::string out6 = "shi_tomasi_outputs/refreshed_from_large_set.png";
    if (!cv::imwrite(out6, vis_refresh_large))
    {
        std::cerr << "Failed to write " << out6 << "\n";
    }

    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Saved:\n"
              << "  " << out1 << " (mine pts=" << pts_mine.size() << ")\n"
              << "  " << out2 << " (opencv pts=" << pts_cv.size() << ")\n"
              << "  " << out3 << " (masked pts=" << pts_masked.size() << ")\n"
              << "  " << out4 << " (mask overlay)\n"
              << "  " << out5 << " (refreshed from 50 -> " << refreshedSmall.size() << ")\n"
              << "  " << out6 << " (refreshed from 250 -> " << refreshedLarge.size() << ")\n";

    std::cout << "\nMask test:\n";
    std::cout << "  Forbidden-area points found: " << forbidden_count << "\n";
    if (forbidden_count == 0)
    {
        std::cout << "  OK: masked detector does not place points in forbidden area.\n";
    }
    else
    {
        std::cout << "  WARNING: some points appeared in forbidden area.\n";
    }

    std::cout << "\nRefresh test:\n";
    std::cout << "  small tracked set:  " << trackedSmall.size()
              << " -> " << refreshedSmall.size() << "\n";
    std::cout << "  large tracked set:  " << trackedLarge.size()
              << " -> " << refreshedLarge.size() << "\n";

    if (refreshedSmall.size() > trackedSmall.size())
    {
        std::cout << "  OK: refresh triggered for small feature set.\n";
    }
    else
    {
        std::cout << "  WARNING: refresh did not add points for small set.\n";
    }

    if (refreshedLarge.size() == trackedLarge.size())
    {
        std::cout << "  OK: refresh skipped for sufficiently large feature set.\n";
    }
    else
    {
        std::cout << "  WARNING: refresh changed large set unexpectedly.\n";
    }

    return 0;
}