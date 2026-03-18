#include <opencv2/opencv.hpp>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Bilinear = двовимірна лінійна інтерполяція.
static float getPixelBilinear(const cv::Mat& img, const float x, const float y)
{
    if (img.type() != CV_32FC1)
        throw std::runtime_error("getPixelBilinear: image must be CV_32FC1");

    if (x < 0.0f || y < 0.0f || x >= img.cols - 1 || y >= img.rows - 1)
        return 0.0f;

    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float a = x - x0;
    const float b = y - y0;

    const float I00 = img.at<float>(y0, x0);
    const float I10 = img.at<float>(y0, x1);
    const float I01 = img.at<float>(y1, x0);
    const float I11 = img.at<float>(y1, x1);

    const float value =
        (1.0f - a) * (1.0f - b) * I00 +
        a * (1.0f - b) * I10 +
        (1.0f - a) * b * I01 +
        a * b * I11;

    return value;
}

// щоб був менший residual в основному з [0,255] -> [0,1]
static cv::Mat toFloatGray(const cv::Mat& gray) {
    if (gray.empty()) throw std::runtime_error("toFloatGray: empty image");
    if (gray.channels() != 1) throw std::runtime_error("toFloatGray: expected 1-channel image");

    cv::Mat f;
    if (gray.type() == CV_32FC1) {
        f = gray.clone();
    } else if (gray.type() == CV_8UC1) {
        gray.convertTo(f, CV_32FC1, 1.0 / 255.0);
    } else {
        gray.convertTo(f, CV_32FC1);
    }
    return f;
}

static int border_replicate(const int val, const int min, const int max)
{
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

static cv::Mat gaussianBlurCustomFloat(const cv::Mat& image)
{
    if (image.empty())
        throw std::runtime_error("gaussianBlurCustomFloat: empty image");
    if (image.type() != CV_32FC1)
        throw std::runtime_error("gaussianBlurCustomFloat: expected CV_32FC1");

    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    cv::Mat result(image.rows, image.cols, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            float sum = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const int yy = border_replicate(y + ky, 0, image.rows - 1);
                    const int xx = border_replicate(x + kx, 0, image.cols - 1);
                    const float pixel = image.at<float>(yy, xx);
                    const int coeff = kernel[ky + 1][kx + 1];
                    sum += pixel * static_cast<float>(coeff);
                }
            }
            result.at<float>(y, x) = sum / 16.0f;
        }
    }
    return result;
}

// Sobel 3x3 gradients for CV_32FC1
static void computeGradients(const cv::Mat& imgGrayF, cv::Mat& Ix, cv::Mat& Iy)
{
    if (imgGrayF.empty()) throw std::runtime_error("computeGradients: empty image");
    if (imgGrayF.type() != CV_32FC1) throw std::runtime_error("computeGradients: expected CV_32FC1");
    if (imgGrayF.channels() != 1) throw std::runtime_error("computeGradients: expected 1-channel image");
    const int rows = imgGrayF.rows;
    const int cols = imgGrayF.cols;
    Ix.create(rows, cols, CV_32FC1);
    Iy.create(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        float* outX = Ix.ptr<float>(y);
        float* outY = Iy.ptr<float>(y);

        for (int x = 0; x < cols; ++x) {
            const int xm1 = border_replicate(x - 1, 0, cols - 1);
            const int xp1 = border_replicate(x + 1, 0, cols - 1);
            const int ym1 = border_replicate(y - 1, 0, rows - 1);
            const int yp1 = border_replicate(y + 1, 0, rows - 1);

            const float a00 = imgGrayF.at<float>(ym1, xm1);
            const float a01 = imgGrayF.at<float>(ym1, x);
            const float a02 = imgGrayF.at<float>(ym1, xp1);

            const float a10 = imgGrayF.at<float>(y, xm1);
            const float a11 = imgGrayF.at<float>(y, x);
            const float a12 = imgGrayF.at<float>(y, xp1);

            const float a20 = imgGrayF.at<float>(yp1, xm1);
            const float a21 = imgGrayF.at<float>(yp1, x);
            const float a22 = imgGrayF.at<float>(yp1, xp1);

            const float Gx =
                (-1.f)*a00 + (0.f)*a01 + (1.f)*a02 +
                (-2.f)*a10 + (0.f)*a11 + (2.f)*a12 +
                (-1.f)*a20 + (0.f)*a21 + (1.f)*a22;
            const float Gy =
                (-1.f)*a00 + (-2.f)*a01 + (-1.f)*a02 +
                 (0.f)*a10 + (0.f)*a11 + (0.f)*a12 +
                 (1.f)*a20 + (2.f)*a21 + (1.f)*a22;
            outX[x] = Gx;
            outY[x] = Gy;
        }
    }
}

// Витягує квадратний patch (2r+1)x(2r+1) навколо (cx,cy). false, якщо patch не влазить (з урахуванням bilinear).
static bool extractPatch(const cv::Mat& imgF, const float cx, const float cy, const int r, cv::Mat& patch)
{
    if (imgF.empty())
        throw std::runtime_error("extractPatch: empty image");
    if (imgF.type() != CV_32FC1 || imgF.channels() != 1)
        throw std::runtime_error("extractPatch: expected CV_32FC1 (1-channel)");
    if (r <= 0)
        throw std::runtime_error("extractPatch: radius r must be > 0");

    // нам потрібні координати для bilinear: x < w-1, y < h-1
    if (cx - r < 0.0f || cy - r < 0.0f || cx + r >= static_cast<float>(imgF.cols - 1)
        || cy + r >= static_cast<float>(imgF.rows - 1)) return false;

    const int size = 2 * r + 1;
    patch.create(size, size, CV_32FC1);
    // patch(y,x) відповідає точці (cx + (x-r), cy + (y-r)) у зображенні
    for (int py = 0; py < size; ++py) {
        float* row = patch.ptr<float>(py);
        const float img_y = cy + (py - r);
        for (int px = 0; px < size; ++px) {
            const float img_x = cx + (px - r);
            row[px] = getPixelBilinear(imgF, img_x, img_y);
        }
    }
    return true;
}

// Обчислює H (2x2), b (2x1) для translation LK на одному рівні. false, якщо ми виходимо за межі (неможливо білінійно семплити).
static bool computeLKSystemTranslation(
    const cv::Mat& patchT,
    const cv::Mat& imgCurrF,
    const cv::Mat& IxCurrF,
    const cv::Mat& IyCurrF,
    const cv::Point2f& p0,
    const cv::Point2f& d,
    const int r,
    cv::Matx22f& H,
    cv::Vec2f& b,
    float& sse
) {
    if (patchT.empty() || imgCurrF.empty() || IxCurrF.empty() || IyCurrF.empty())
        throw std::runtime_error("computeLKSystemTranslation: empty input");
    if (patchT.type() != CV_32FC1 || imgCurrF.type() != CV_32FC1 ||
        IxCurrF.type() != CV_32FC1 || IyCurrF.type() != CV_32FC1)
        throw std::runtime_error("computeLKSystemTranslation: expected CV_32FC1 inputs");
    if (patchT.rows != 2 * r + 1 || patchT.cols != 2 * r + 1)
        throw std::runtime_error("computeLKSystemTranslation: patchT size != (2r+1)x(2r+1)");
    if (imgCurrF.size() != IxCurrF.size() || imgCurrF.size() != IyCurrF.size())
        throw std::runtime_error("computeLKSystemTranslation: gradients size mismatch");

    // Перевірка меж. Найдальші offset +/- r
    const float cx = p0.x + d.x;
    const float cy = p0.y + d.y;

    if (cx - r < 0.0f || cy - r < 0.0f || cx + r >= static_cast<float>(imgCurrF.cols - 1)
        || cy + r >= static_cast<float>(imgCurrF.rows - 1)) return false;
    H = cv::Matx22f::zeros();
    b = cv::Vec2f(0.f, 0.f);
    sse = 0.0f;

    for (int py = 0; py < patchT.rows; ++py) {
        const float* tRow = patchT.ptr<float>(py);
        const float offset_y = static_cast<float>(py - r);
        for (int px = 0; px < patchT.cols; ++px) {
            const float offset_x = static_cast<float>(px - r);
            const float x = cx + offset_x;
            const float y = cy + offset_y;
            const float I = getPixelBilinear(imgCurrF, x, y);
            const float gx = getPixelBilinear(IxCurrF, x, y);
            const float gy = getPixelBilinear(IyCurrF, x, y);
            const float T = tRow[px];
            const float residual = I - T;
            H(0,0) += gx * gx;
            H(0,1) += gx * gy;
            H(1,0) += gx * gy;
            H(1,1) += gy * gy;
            b[0] += gx * residual;
            b[1] += gy * residual;
            sse += residual * residual;
        }
    }
    return true;
}

static bool solve2x2(const cv::Matx22f& H,const cv::Vec2f& val, cv::Vec2f& delta, const float minDet = 1e-6f)
{
    const float a = H(0,0);
    const float b = H(0,1);
    const float c = H(1,0);
    const float d = H(1,1);
    const float det = a * d - b * c;
    if (std::abs(det) < minDet)
        return false;
    const float invDet = 1.0f / det;
    const cv::Matx22f H_inv(
         d * invDet,   -b * invDet,
        -c * invDet,    a * invDet
    );
    delta = H_inv * val;
    return true;
}

static bool trackPointOneLevel(
    const cv::Mat& imgPrevF,
    const cv::Mat& imgCurrF,
    const cv::Mat& IxCurrF,
    const cv::Mat& IyCurrF,
    const cv::Point2f& ptPrev,
    cv::Point2f& ptCurr,
    const int winSize,
    const int maxIters,
    const float eps,
    float& finalErr
) {
    const int r = winSize / 2;
    if (winSize <= 1 || (winSize % 2) == 0)
        throw std::runtime_error("trackPointOneLevel: winSize must be odd and > 1");
    if (maxIters <= 0)
        throw std::runtime_error("trackPointOneLevel: maxIters must be > 0");
    if (eps <= 0.0f)
        throw std::runtime_error("trackPointOneLevel: eps must be > 0");
    cv::Mat patchT;
    if (!extractPatch(imgPrevF, ptPrev.x, ptPrev.y, r, patchT)) {
        finalErr = -1.0f;
        return false;
    }
    cv::Point2f d = ptCurr - ptPrev;
    bool success = false;
    float sse = 0.0f;
    for (int iter = 0; iter < maxIters; ++iter) {
        cv::Matx22f H;
        cv::Vec2f b;
        sse = 0.0f;
        if (!computeLKSystemTranslation(patchT, imgCurrF, IxCurrF, IyCurrF, ptPrev, d, r, H, b, sse)) {
            finalErr = -1.0f;
            return false;
        }
        cv::Vec2f delta;
        if (!solve2x2(H, -b, delta)) {
            finalErr = -1.0f;
            return false;
        }
        d.x += delta[0];
        d.y += delta[1];
        const float deltaNorm2 = delta[0] * delta[0] + delta[1] * delta[1];
        success = true;
        if (deltaNorm2 < eps * eps)
            break;
    }
    ptCurr = ptPrev + d;
    finalErr = sse / static_cast<float>(winSize * winSize);
    return success;
}

static void trackPointsLKOneLevel(
    const cv::Mat& imgPrevGray,
    const cv::Mat& imgCurrGray,
    const std::vector<cv::Point2f>& pts0,
    std::vector<cv::Point2f>& pts1,
    std::vector<uchar>& status,
    std::vector<float>& err,
    const int winSize = 9,
    const int maxIters = 10,
    const float eps = 1e-3f
) {
    if (imgPrevGray.empty() || imgCurrGray.empty())
        throw std::runtime_error("trackPointsLKOneLevel: empty input image");

    const cv::Mat prevF = toFloatGray(imgPrevGray);
    const cv::Mat currF = toFloatGray(imgCurrGray);

    cv::Mat IxCurr, IyCurr;
    computeGradients(currF, IxCurr, IyCurr);

    pts1.resize(pts0.size());
    status.assign(pts0.size(), 0);
    err.assign(pts0.size(), -1.0f);

    for (size_t i = 0; i < pts0.size(); ++i) {
        cv::Point2f trackedPt = pts0[i];
        float trackErr = -1.0f;

        const bool ok = trackPointOneLevel(
            prevF,
            currF,
            IxCurr,
            IyCurr,
            pts0[i],
            trackedPt,
            winSize,
            maxIters,
            eps,
            trackErr
        );

        pts1[i] = trackedPt;
        status[i] = ok ? 1 : 0;
        err[i] = trackErr;
    }
}


int main() {
    cv::VideoCapture cap("../IMG_4273.MOV");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video source\n";
        return 1;
    }

    cv::Mat prevFrame;
    if (!cap.read(prevFrame) || prevFrame.empty()) {
        std::cerr << "Failed to read first frame from video." << std::endl;
        return 1;
    }

    cv::Mat prevGray;
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> ptsPrev;
    cv::goodFeaturesToTrack(prevGray, ptsPrev, 300, 0.01, 10.0);

    if (ptsPrev.empty()) {
        std::cerr << "No features found in the first frame." << std::endl;
        return 1;
    }

    while (true) {
        cv::Mat currFrame;
        if (!cap.read(currFrame) || currFrame.empty()) {
            break;
        }

        cv::Mat currGray;
        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> ptsCurr;
        std::vector<uchar> status;
        std::vector<float> err;

        try {
            auto t1 = std::chrono::high_resolution_clock::now();

            trackPointsLKOneLevel(
                prevGray,
                currGray,
                ptsPrev,
                ptsCurr,
                status,
                err,
                9,
                10,
                1e-3f
            );

            auto t2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            std::cout << "Tracking time: " << ms << " ms" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Tracking error: " << e.what() << std::endl;
            return 1;
        }

        cv::Mat vis = currFrame.clone();

        std::vector<cv::Point2f> goodCurr;
        for (size_t i = 0; i < ptsPrev.size(); ++i) {
            if (!status[i]) continue;

            const cv::Point2f& p0 = ptsPrev[i];
            const cv::Point2f& p1 = ptsCurr[i];

            cv::circle(vis, p1, 3, cv::Scalar(0, 255, 0), -1);
            cv::line(vis, p0, p1, cv::Scalar(0, 255, 0), 1);

            goodCurr.push_back(p1);
        }

        cv::imshow("One-Level LK Tracking", vis);
        cv::waitKey(1);

        if (goodCurr.size() < 50) {
            cv::goodFeaturesToTrack(currGray, goodCurr, 300, 0.01, 10.0);
        }

        prevGray = currGray.clone();
        ptsPrev = goodCurr;

        if (ptsPrev.empty()) {
            std::cerr << "No points left to track." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}