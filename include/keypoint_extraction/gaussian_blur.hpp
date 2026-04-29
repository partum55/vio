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

void gaussianBlurCustomBanded(
    const cv::Mat& image,
    cv::Mat& result,
    ABCThreadPool& pool,
    int num_threads,
    int rows_per_chunk = 64);

cv::Mat gaussianBlurCustomBanded(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_threads,
    int rows_per_chunk = 64);