#include "keypoints/gaussian_blur.hpp"
#include "keypoints/parallel_utils.hpp"

#include <algorithm>

namespace vio {

cv::Mat convolveHorizontalReplicate(
    const cv::Mat& image,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks) {
    CV_Assert(image.type() == CV_32F);
    CV_Assert(image.cols >= 2);

    cv::Mat result(image.rows, image.cols, CV_32F);

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                const float* src = image.ptr<float>(y);
                float* dst = result.ptr<float>(y);

                dst[0] = kernel[0] * src[0] +
                         kernel[1] * src[0] +
                         kernel[2] * src[1];

                for (int x = 1; x < image.cols - 1; ++x) {
                    dst[x] = kernel[0] * src[x - 1] +
                             kernel[1] * src[x] +
                             kernel[2] * src[x + 1];
                }

                const int last = image.cols - 1;
                dst[last] = kernel[0] * src[last - 1] +
                            kernel[1] * src[last] +
                            kernel[2] * src[last];
            }
        });

    return result;
}

cv::Mat convolveVerticalReplicate(
    const cv::Mat& image,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks) {
    CV_Assert(image.type() == CV_32F);

    cv::Mat result(image.rows, image.cols, CV_32F);

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                float* dst = result.ptr<float>(y);

                const float* row_up = image.ptr<float>(std::max(0, y - 1));
                const float* row_mid = image.ptr<float>(y);
                const float* row_down = image.ptr<float>(std::min(image.rows - 1, y + 1));

                for (int x = 0; x < image.cols; ++x) {
                    dst[x] = kernel[0] * row_up[x] +
                             kernel[1] * row_mid[x] +
                             kernel[2] * row_down[x];
                }
            }
        });

    return result;
}

cv::Mat gaussianBlurCustom(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks) {
    CV_Assert(image.type() == CV_32F);

    constexpr float kernel[3] = {0.25f, 0.5f, 0.25f};

    cv::Mat temp = convolveHorizontalReplicate(image, kernel, pool, num_tasks);
    return convolveVerticalReplicate(temp, kernel, pool, num_tasks);
}

} // namespace vio
