#include <opencv2/opencv.hpp>
#include <iostream>
// глянути як оптимізувати щоб воно швилко йшло
//  отримуєм стартові keypoints (у форматі KeyPoint)
static std::vector<cv::KeyPoint> loadInitialKeypoints() {
    std::vector<cv::KeyPoint> kps;
    //просто приклад
    kps.emplace_back(cv::Point2f(150.f, 120.f), 1.f);
    kps.emplace_back(cv::Point2f(300.f, 200.f), 1.f);
    kps.emplace_back(cv::Point2f(400.f, 250.f), 1.f);

    return kps;
}

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
    cv::VideoCapture cap("../IMG_4273.MOV");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video source\n";
        return 1;
    }
    const double SCALE = 0.5;        // iPhone відео часто велике: 0.5 або 0.33
    const int MAX_CORNERS = 800;
    const int MIN_TRACKS = 250;      // якщо треків менше — додати нові

    // структура, яка зберігає пікселі зображення.(матриці)
    cv::Mat prev, curr, prev_gray, curr_gray;

    cap >> prev;
    if (prev.empty()) return 0;
    if (SCALE != 1.0) cv::resize(prev, prev, cv::Size(), SCALE, SCALE);
    cv::cvtColor(prev, prev_gray, cv::COLOR_BGR2GRAY);
    // std::cout << "Frame size: " << prev.cols << "x" << prev.rows << "\n";

    // стартові KeyPoints
    // std::vector<cv::KeyPoint> initialKps = loadInitialKeypoints();

    // стартові точки
    std::vector<cv::Point2f> pts0;
    detectPoints(prev_gray, pts0, MAX_CORNERS);
    if (pts0.empty()) {
        std::cerr << "Initial keypoints are empty.\n";
        return 1;
    }

    //Конвертуємо KeyPoint в Point2f
    // std::vector<cv::Point2f> pts0;
    // cv::KeyPoint::convert(initialKps, pts0);

    // LK
    cv::Size winSize(21, 21);

    // це короче для піраміди зменшуємо зображення у степіь двійки
    int maxLevel = 3;

    // критерій зупинки ітерацій
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

    while (true) {
        cap >> curr;
        if (curr.empty()) break;

        if (SCALE != 1.0) cv::resize(curr, curr, cv::Size(), SCALE, SCALE);
        cv::cvtColor(curr, curr_gray, cv::COLOR_BGR2GRAY);

        // Lucas–Kanade tracking
        std::vector<cv::Point2f> pts1;
        std::vector<uchar> status; // вдалось не вдалось знайти точку
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            pts0, pts1,
            status, err,
            winSize, maxLevel, criteria
        );

        // Фільтрація по status
        std::vector<cv::Point2f> good0, good1;
        good0.reserve(pts0.size());
        good1.reserve(pts0.size());

        for (size_t i = 0; i < pts0.size(); ++i) {
            if (!status[i]) continue;
            good0.push_back(pts0[i]);
            good1.push_back(pts1[i]);
        }
        if (good1.empty()) {
            std::cerr << "All tracks lost.\n";
            break;
        }

        // Візуалізація
        cv::Mat vis = curr.clone();
        for (size_t i = 0; i < good1.size(); ++i) {
            cv::circle(vis, good1[i], 3, cv::Scalar(0, 255, 0), -1);
            cv::line(vis, good0[i], good1[i], cv::Scalar(0, 255, 0), 1);
        }

        cv::putText(vis,
                    "tracks: " + std::to_string(good1.size()),
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(255, 255, 255),
                    2);

        cv::imshow("KLT Tracking", vis);

        // Оновлення для наступної ітерації
        pts0 = good1;
        prev_gray = curr_gray.clone();
        prev = curr.clone();


        if (cv::waitKey(1) == 27) break; // натиснути ESC щоб примусово вийти з циклу
    }
    return 0;
}
