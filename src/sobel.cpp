#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <filesystem>

#include "gaussian_blur.hpp"

using namespace cv;
using namespace std;

static int clampInt(int value, int low, int high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static Mat toGrayscale8(const Mat& image) {
    Mat gray;

    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cvtColor(image, gray, COLOR_BGRA2GRAY);
    } else {
        gray = image.clone();
    }

    if (gray.type() != CV_8U) {
        Mat converted;
        gray.convertTo(converted, CV_8U);
        return converted;
    }

    return gray;
}

static Mat convolveReplicateBorder(const Mat& image, const int kernel[3][3]) {
    CV_Assert(image.type() == CV_8U);

    Mat result(image.rows, image.cols, CV_32F, Scalar(0));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            float sum = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int yy = clampInt(y + ky, 0, image.rows - 1);
                    int xx = clampInt(x + kx, 0, image.cols - 1);

                    int pixel = static_cast<int>(image.at<uchar>(yy, xx));
                    int coeff = kernel[ky + 1][kx + 1];

                    sum += static_cast<float>(pixel * coeff);
                }
            }

            result.at<float>(y, x) = sum;
        }
    }

    return result;
}

static Mat normalizeForDisplaySigned(const Mat& src32f) {
    CV_Assert(src32f.type() == CV_32F);

    double minVal = 0.0;
    double maxVal = 0.0;
    minMaxLoc(src32f, &minVal, &maxVal);

    double maxAbs = max(std::abs(minVal), std::abs(maxVal));

    if (maxAbs < 1e-6) {
        return Mat(src32f.size(), CV_8U, Scalar(128));
    }

    Mat display(src32f.rows, src32f.cols, CV_8U);

    for (int y = 0; y < src32f.rows; ++y) {
        for (int x = 0; x < src32f.cols; ++x) {
            float v = src32f.at<float>(y, x);

            float normalized = 127.5f + 127.5f * (v / static_cast<float>(maxAbs));
            normalized = std::max(0.0f, std::min(255.0f, normalized));

            display.at<uchar>(y, x) = static_cast<uchar>(normalized);
        }
    }

    return display;
}

static Mat normalizeForDisplayUnsigned(const Mat& src32f) {
    CV_Assert(src32f.type() == CV_32F);

    double minVal = 0.0;
    double maxVal = 0.0;
    minMaxLoc(src32f, &minVal, &maxVal);

    if (maxVal - minVal < 1e-6) {
        return Mat(src32f.size(), CV_8U, Scalar(0));
    }

    Mat display;
    src32f.convertTo(
        display,
        CV_8U,
        255.0 / (maxVal - minVal),
        -minVal * 255.0 / (maxVal - minVal)
    );

    return display;
}

static Mat computeGradientMagnitude(const Mat& gx, const Mat& gy) {
    CV_Assert(gx.type() == CV_32F && gy.type() == CV_32F);
    CV_Assert(gx.size() == gy.size());

    Mat magnitude(gx.rows, gx.cols, CV_32F, Scalar(0));

    for (int y = 0; y < gx.rows; ++y) {
        for (int x = 0; x < gx.cols; ++x) {
            float dx = gx.at<float>(y, x);
            float dy = gy.at<float>(y, x);

            magnitude.at<float>(y, x) = std::sqrt(dx * dx + dy * dy);
        }
    }

    return magnitude;
}

static Mat thresholdMagnitude(const Mat& magnitude32f, float thresholdValue) {
    CV_Assert(magnitude32f.type() == CV_32F);

    Mat edges(magnitude32f.rows, magnitude32f.cols, CV_8U, Scalar(0));

    for (int y = 0; y < magnitude32f.rows; ++y) {
        for (int x = 0; x < magnitude32f.cols; ++x) {
            float value = magnitude32f.at<float>(y, x);
            edges.at<uchar>(y, x) = (value >= thresholdValue) ? 255 : 0;
        }
    }

    return edges;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./sobel_demo <image_path>" << endl;
        return 1;
    }

    string imagePath = argv[1];
    Mat input = imread(imagePath, IMREAD_UNCHANGED);

    if (input.empty()) {
        cerr << "Failed to load image: " << imagePath << endl;
        return 1;
    }

    Mat gray = toGrayscale8(input);

    Mat blurred = gaussianBlurCustom(gray);

    const int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int sobelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    Mat gx = convolveReplicateBorder(blurred, sobelX);
    Mat gy = convolveReplicateBorder(blurred, sobelY);

    Mat magnitude = computeGradientMagnitude(gx, gy);

    Mat gxVis = normalizeForDisplaySigned(gx);
    Mat gyVis = normalizeForDisplaySigned(gy);
    Mat magVis = normalizeForDisplayUnsigned(magnitude);

    float thresholdValue = 100.0f;
    Mat edges = thresholdMagnitude(magnitude, thresholdValue);

    std::filesystem::create_directories("sobel");

    if (!imwrite("sobel/01_gray.png", gray))
        cerr << "Failed to write sobel/01_gray.png" << endl;
    if (!imwrite("sobel/02_blurred.png", blurred))
        cerr << "Failed to write sobel/02_blurred.png" << endl;
    if (!imwrite("sobel/03_gx.png", gxVis))
        cerr << "Failed to write sobel/03_gx.png" << endl;
    if (!imwrite("sobel/04_gy.png", gyVis))
        cerr << "Failed to write sobel/04_gy.png" << endl;
    if (!imwrite("sobel/05_magnitude.png", magVis))
        cerr << "Failed to write sobel/05_magnitude.png" << endl;
    if (!imwrite("sobel/06_edges.png", edges))
        cerr << "Failed to write sobel/06_edges.png" << endl;

    cout << "Finished!" << endl;
    return 0;
}