// Будує image pyramid для prev і curr
// Для кожної точки:
//     бере patch (winSize × winSize)
//     рахує градієнти
//     формує Jacobian
//     мінімізує photometric error (SSD)
//     ітеративно оновлює зміщення (Gauss–Newton)
// Повертає:
//     нові координати pts1
//     status (вдалось / ні)
//     err (residual error)

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <stdexcept>
#include <iostream>

// Bilinear = двовимірна лінійна інтерполяція.
static float getPixelBilinear(const cv::Mat& img, const float x, const float y)
{
    if (img.type() != CV_32FC1)
        throw std::runtime_error("getPixelBilinear: image must be CV_32FC1");

    if (x < 0.0f || y < 0.0f || x >= img.cols - 1 || y >= img.rows - 1)
        return 0.0f;

    // Ліва верхня ціла координата
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    // Дробові частини
    const float a = x - x0;
    const float b = y - y0;

    // Доступ до пікселів
    const float I00 = img.at<float>(y0, x0);
    const float I10 = img.at<float>(y0, x1);
    const float I01 = img.at<float>(y1, x0);
    const float I11 = img.at<float>(y1, x1);

    // Bilinear interpolation
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

static std::vector<cv::Mat> buildPyramid(const cv::Mat& imgGray, const int levels)
{
    std::vector<cv::Mat> pyr;
    pyr.reserve(levels + 1);
    const cv::Mat cur = toFloatGray(imgGray);
    pyr.push_back(cur);
    for (int i = 1; i <= levels; i++)
    {
        cv::Mat prev;
        prev = pyr[i-1];
        cv::Mat blurred = gaussianBlur(prev, kernel=5x5, sigma=1.0);
        int newWidth  = floor(prev.cols / 2);
        int newHeight = floor(prev.rows / 2);
        cv::Mat curr = image(newHeight, newWidth);
        for (int y = 0; y < newHeight; y++)
            for (int x = 0; x < newWidth; x++)
            {
                curr[y, x] = blurred[2*y, 2*x];
            }
        pyr[i] = curr;
    }
    return pyr;
}

static int border_replicate(const int val, const int min, const int max)
{
    if (val < min) return min;
    if (val > max) return max;
    return val;
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

    // Sobel kernels:
    // Gx = [ -1  0  +1
    //        -2  0  +2
    //        -1  0  +1 ]
    //
    // Gy = [ -1 -2 -1
    //         0  0  0
    //        +1 +2 +1 ]

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

            const float a10 = imgGrayF.at<float>(y,   xm1);
            const float a11 = imgGrayF.at<float>(y,   x);
            const float a12 = imgGrayF.at<float>(y,   xp1);

            const float a20 = imgGrayF.at<float>(yp1, xm1);
            const float a21 = imgGrayF.at<float>(yp1, x);
            const float a22 = imgGrayF.at<float>(yp1, xp1);

            const float Gx =
                (-1.f)*a00 + (0.f)*a01 + (1.f)*a02 +
                (-2.f)*a10 + (0.f)*a11 + (2.f)*a12 +
                (-1.f)*a20 + (0.f)*a21 + (1.f)*a22;

            const float Gy =
                (-1.f)*a00 + (-2.f)*a01 + (-1.f)*a02 +
                 (0.f)*a10 +  (0.f)*a11 +  (0.f)*a12 +
                 (1.f)*a20 +  (2.f)*a21 +  (1.f)*a22;

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

// computeLKSystemTranslation

static bool solve2x2(const cv::Matx22f& H,const cv::Vec2f& val, cv::Vec2f& delta, const float minDet = 1e-6f)
{
    // H = [ a  b ]
    //     [ c  d ]
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

// trackPointOneLevel
//
// trackPointPyramidal
//
// trackPointsPyramidalLK