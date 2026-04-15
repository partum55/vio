#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

float getPixelBilinear(const cv::Mat& img, const float x, const float y);

cv::Mat toFloatGray(const cv::Mat& gray);

int border_replicate(const int val, const int min, const int max);

cv::Mat gaussianBlurCustomFloat(const cv::Mat& image);

std::vector<cv::Mat> buildPyramid(const cv::Mat& imgGray, const int levels);

void computeGradients(const cv::Mat& imgGrayF, cv::Mat& Ix, cv::Mat& Iy);

bool extractPatch(const cv::Mat& imgF, const float cx, const float cy, const int r, cv::Mat& patch);

bool computeLKSystemTranslation(
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
);

bool solve2x2(const cv::Matx22f& H,const cv::Vec2f& val, cv::Vec2f& delta, const float minDet = 1e-6f);

bool trackPointOneLevel(
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
);

bool trackPointPyramidal(
    const std::vector<cv::Mat>& pyrPrev,
    const std::vector<cv::Mat>& pyrCurr,
    const std::vector<cv::Mat>& pyrIx,
    const std::vector<cv::Mat>& pyrIy,
    const cv::Point2f& ptPrev,
    cv::Point2f& ptCurr,
    const int winSize,
    const int maxIters,
    const float eps,
    float& finalErr
);

void trackPointsPyramidalLK(
    const cv::Mat& imgPrevGray,
    const cv::Mat& imgCurrGray,
    const std::vector<cv::Point2f>& pts0,
    std::vector<cv::Point2f>& pts1,
    std::vector<uchar>& status,
    std::vector<float>& err,
    const int winSize = 9,
    const int maxLevel = 3,
    const int maxIters = 10,
    const float eps = 1e-3f
);
