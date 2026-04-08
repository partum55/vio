#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

float getPixelBilinear(const cv::Mat& img, float x, float y);
cv::Mat toFloatGray(const cv::Mat& gray);
int border_replicate(int val, int min, int max);

void computeGradients(const cv::Mat& imgGrayF, cv::Mat& Ix, cv::Mat& Iy);

bool extractPatch(const cv::Mat& imgF, float cx, float cy, int r, cv::Mat& patch);

bool computeLKSystemTranslation(
    const cv::Mat& patchT,
    const cv::Mat& imgCurrF,
    const cv::Mat& IxCurrF,
    const cv::Mat& IyCurrF,
    const cv::Point2f& p0,
    const cv::Point2f& d,
    int r,
    cv::Matx22f& H,
    cv::Vec2f& b,
    float& sse
);

bool solve2x2(
    const cv::Matx22f& H,
    const cv::Vec2f& val,
    cv::Vec2f& delta,
    float minDet = 1e-6f
);

bool trackPointOneLevel(
    const cv::Mat& imgPrevF,
    const cv::Mat& imgCurrF,
    const cv::Mat& IxCurrF,
    const cv::Mat& IyCurrF,
    const cv::Point2f& ptPrev,
    cv::Point2f& ptCurr,
    int winSize,
    int maxIters,
    float eps,
    float& finalErr
);

void trackPointsLKSingleLevel(
    const cv::Mat& imgPrevGray,
    const cv::Mat& imgCurrGray,
    const std::vector<cv::Point2f>& pts0,
    std::vector<cv::Point2f>& pts1,
    std::vector<uchar>& status,
    std::vector<float>& err,
    int winSize = 9,
    int maxIters = 10,
    float eps = 1e-3f
);