#include "gaussian_blur.hpp"

using namespace cv;

static int clampInt(int value, int low, int high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

Mat gaussianBlurCustom(const Mat& image) {
    CV_Assert(image.type() == CV_8U);

    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    Mat result(image.rows, image.cols, CV_8U, Scalar(0));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int sum = 0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int yy = clampInt(y + ky, 0, image.rows - 1);
                    int xx = clampInt(x + kx, 0, image.cols - 1);

                    int pixel = static_cast<int>(image.at<uchar>(yy, xx));
                    int coeff = kernel[ky + 1][kx + 1];

                    sum += pixel * coeff;
                }
            }

            result.at<uchar>(y, x) = static_cast<uchar>(sum / 16);
        }
    }

    return result;
}
