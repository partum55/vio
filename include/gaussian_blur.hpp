#pragma once

#include "abc_thread_pool.hpp"

#include <opencv2/opencv.hpp>

void convolveHorizontalReplicate(
    const cv::Mat& image,
    cv::Mat& result,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks);

void convolveVerticalReplicate(
    const cv::Mat& image,
    cv::Mat& result,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks);

void gaussianBlurCustom(
    const cv::Mat& image,
    cv::Mat& result,
    ABCThreadPool& pool,
    int num_tasks);

cv::Mat gaussianBlurCustom(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks);