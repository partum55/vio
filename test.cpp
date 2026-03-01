#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

static void detectPoints(const cv::Mat& gray,
                         std::vector<cv::Point2f>& pts,
                         int maxCorners = 800,
                         double quality = 0.01,
                         double minDist = 10.0)
{
    // Shi–Tomasi corners
    cv::goodFeaturesToTrack(gray, pts, maxCorners, quality, minDist);
}

int main() {
    std::cout << "Working dir: " << std::filesystem::current_path() << std::endl;
    // Відео лежить поруч з main.cpp / CMakeLists.txt
    cv::VideoCapture cap("../IMG_4273.MOV");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video source (IMG_4272.MOV)\n";
        return 1;
    }

    const double SCALE = 0.5;        // iPhone відео часто велике: 0.5 або 0.33
    const int MAX_CORNERS = 800;
    const int MIN_TRACKS = 250;      // якщо треків менше — додати нові

    cv::Mat prev, curr, prev_gray, curr_gray;

    // --- перший кадр ---
    cap >> prev;
    if (prev.empty()) return 0;

    if (SCALE != 1.0) cv::resize(prev, prev, cv::Size(), SCALE, SCALE);
    cv::cvtColor(prev, prev_gray, cv::COLOR_BGR2GRAY);

    std::cout << "Frame size (after scale): " << prev.cols << "x" << prev.rows << "\n";

    // --- стартові точки (автоматично) ---
    std::vector<cv::Point2f> pts0;
    detectPoints(prev_gray, pts0, MAX_CORNERS);

    if (pts0.empty()) {
        std::cerr << "No points detected on the first frame.\n";
        return 1;
    }

    // --- LK parameters ---
    cv::Size winSize(21, 21);
    int maxLevel = 3;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

    while (true) {
        cap >> curr;
        if (curr.empty()) break;

        if (SCALE != 1.0) cv::resize(curr, curr, cv::Size(), SCALE, SCALE);
        cv::cvtColor(curr, curr_gray, cv::COLOR_BGR2GRAY);

        // 1) LK tracking
        std::vector<cv::Point2f> pts1;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            pts0, pts1,
            status, err,
            winSize, maxLevel, criteria
        );

        // 2) Filter by status
        std::vector<cv::Point2f> good0, good1;
        good0.reserve(pts0.size());
        good1.reserve(pts0.size());

        for (size_t i = 0; i < pts0.size(); ++i) {
            if (!status[i]) continue;
            good0.push_back(pts0[i]);
            good1.push_back(pts1[i]);
        }

        // 3) Якщо треків мало — додати нові точки (на поточному кадрі)
        if ((int)good1.size() < MIN_TRACKS) {
            std::vector<cv::Point2f> newPts;
            detectPoints(curr_gray, newPts, MAX_CORNERS);

            // Простий варіант: замінити на нові (стабільно для демо)
            // Для продакшену можна робити merge + маску, але поки не потрібно.
            good1 = newPts;
        }

        if (good1.empty()) {
            std::cerr << "All tracks lost.\n";
            break;
        }

        // 4) Visualization
        cv::Mat vis = curr.clone();
        for (size_t i = 0; i < good1.size() && i < good0.size(); ++i) {
            cv::circle(vis, good1[i], 3, cv::Scalar(0, 255, 0), -1);
            cv::line(vis, good0[i], good1[i], cv::Scalar(0, 255, 0), 1);
        }

        cv::putText(vis,
                    "tracks: " + std::to_string(good1.size()) + " (ESC exit, r redetect)",
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(255, 255, 255),
                    2);

        cv::imshow("KLT Tracking", vis);

        // 5) Update state
        pts0 = good1;
        prev_gray = curr_gray.clone();
        prev = curr.clone();

        int key = cv::waitKey(1);
        if (key == 27) break;   // ESC
        if (key == 'r') {
            detectPoints(prev_gray, pts0, MAX_CORNERS);
        }
    }

    return 0;
}