#include "shi_tomasi.hpp"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: compare_keypoints <image_path>\n";
        return 1;
    }

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

    std::vector<cv::Point2f> pts_mine = extractShiTomasiKeypoints(img, p);

    // OpenCV
    cv::Mat gray = toGrayU8(img);

    std::vector<cv::Point2f> pts_cv;
    cv::goodFeaturesToTrack(
        gray,
        pts_cv,
        p.maxCorners,
        p.qualityLevel,
        p.minDistance,
        cv::noArray(),
        p.blockSize,
        false, // useHarrisDetector = false => Shi–Tomasi
        0.04);

    cv::Mat vis_mine = drawKeypointsOnImage(img, pts_mine);
    cv::Mat vis_cv = drawKeypointsOnImage(img, pts_cv);

    cv::putText(vis_mine, "Mine", {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::putText(vis_cv, "OpenCV", {20, 40},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    const std::string out1 = "keypoints_mine.png";
    const std::string out2 = "keypoints_opencv.png";

    if (!cv::imwrite(out1, vis_mine))
        std::cerr << "Failed to write " << out1 << "\n";
    if (!cv::imwrite(out2, vis_cv))
        std::cerr << "Failed to write " << out2 << "\n";

    std::cout << "Saved:\n  " << out1 << " (mine pts=" << pts_mine.size() << ")\n"
              << "  " << out2 << " (opencv pts=" << pts_cv.size() << ")\n";

    return 0;
}
