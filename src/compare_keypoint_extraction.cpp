#include "shi_tomasi.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: compare_keypoints <image_path>\n";
        return 1;
    }

    std::filesystem::create_directories("shi_tomasi_outputs");

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Failed to read image: " << argv[1] << "\n";
        return 1;
    }

    ShiTomasiParams p;
    p.maxCorners = 500;
    p.qualityLevel = 0.01;
    p.minDistance = 10.0;
    p.blockSize = 5;
    p.sobelKSize = 3;
    p.gaussianSigma = 1.0;
    p.nmsRadius = 2;

    CustomShiTomasiDetector myDetector;
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

    std::cout << "Saved:\n"
              << "  " << out1 << " (mine pts=" << pts_mine.size() << ")\n"
              << "  " << out2 << " (opencv pts=" << pts_cv.size() << ")\n";

    return 0;
}
