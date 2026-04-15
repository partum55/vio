#include "keypoints/parallel_utils.hpp"
#include "keypoints/sobel.hpp"

#include <algorithm>

namespace vio {

cv::Mat derivativeXCentral(
    const cv::Mat& image,
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

                dst[0] = 0.5f * (src[1] - src[0]);

                for (int x = 1; x < image.cols - 1; ++x) {
                    dst[x] = 0.5f * (src[x + 1] - src[x - 1]);
                }

                const int last = image.cols - 1;
                dst[last] = 0.5f * (src[last] - src[last - 1]);
            }
        });

    return result;
}

cv::Mat derivativeYCentral(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks) {
    CV_Assert(image.type() == CV_32F);

    cv::Mat result(image.rows, image.cols, CV_32F);

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                float* dst = result.ptr<float>(y);

                const float* row_up = image.ptr<float>(std::max(0, y - 1));
                const float* row_down = image.ptr<float>(std::min(image.rows - 1, y + 1));

                for (int x = 0; x < image.cols; ++x) {
                    dst[x] = 0.5f * (row_down[x] - row_up[x]);
                }
            }
        });

    return result;
}

void centralDifferenceXY(
    const cv::Mat& image,
    cv::Mat& gx,
    cv::Mat& gy,
    ABCThreadPool& pool,
    int num_tasks) {
    CV_Assert(image.type() == CV_32F);
    CV_Assert(image.cols >= 2);

    gx.create(image.size(), CV_32F);
    gy.create(image.size(), CV_32F);

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                const float* row = image.ptr<float>(y);
                const float* up = image.ptr<float>(std::max(0, y - 1));
                const float* dn = image.ptr<float>(std::min(image.rows - 1, y + 1));

                float* dx = gx.ptr<float>(y);
                float* dy = gy.ptr<float>(y);

                dx[0] = 0.5f * (row[1] - row[0]);

                for (int x = 1; x < image.cols - 1; ++x) {
                    dx[x] = 0.5f * (row[x + 1] - row[x - 1]);
                }

                const int last = image.cols - 1;
                dx[last] = 0.5f * (row[last] - row[last - 1]);

                for (int x = 0; x < image.cols; ++x) {
                    dy[x] = 0.5f * (dn[x] - up[x]);
                }
            }
        });
}

} // namespace vio
