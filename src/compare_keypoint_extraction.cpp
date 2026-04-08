#include "shi_tomasi.hpp"
#include "tpool_default.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char **argv)
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

    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Saved:\n"
              << "  " << out1 << " (mine pts=" << pts_mine.size() << ")\n"
              << "  " << out2 << " (opencv pts=" << pts_cv.size() << ")\n";

    return 0;
}
