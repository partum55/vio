#include "sobel.hpp"
#include "parallel_utils.hpp"

#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

cv::Mat derivativeXCentral(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks)
{
    CV_Assert(image.type() == CV_32F);
    CV_Assert(image.cols >= 2);

    cv::Mat result(image.rows, image.cols, CV_32F);

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                const float* src = image.ptr<float>(y);
                float* dst = result.ptr<float>(y);

                dst[0] = 0.5f * (src[1] - src[0]);

                int x = 1;

#if defined(__AVX2__)
                const __m256 half = _mm256_set1_ps(0.5f);
                for (; x + 7 < image.cols - 1; x += 8)
                {
                    const __m256 right = _mm256_loadu_ps(src + x + 1);
                    const __m256 left  = _mm256_loadu_ps(src + x - 1);
                    const __m256 diff  = _mm256_sub_ps(right, left);
                    const __m256 res   = _mm256_mul_ps(diff, half);
                    _mm256_storeu_ps(dst + x, res);
                }
#endif

                for (; x < image.cols - 1; ++x)
                {
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
    int num_tasks)
{
    CV_Assert(image.type() == CV_32F);

    cv::Mat result(image.rows, image.cols, CV_32F);

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                float* dst = result.ptr<float>(y);

                const float* row_up   = image.ptr<float>(std::max(0, y - 1));
                const float* row_down = image.ptr<float>(std::min(image.rows - 1, y + 1));

                int x = 0;

#if defined(__AVX2__)
                const __m256 half = _mm256_set1_ps(0.5f);
                for (; x + 7 < image.cols; x += 8)
                {
                    const __m256 up   = _mm256_loadu_ps(row_up + x);
                    const __m256 down = _mm256_loadu_ps(row_down + x);
                    const __m256 diff = _mm256_sub_ps(down, up);
                    const __m256 res  = _mm256_mul_ps(diff, half);
                    _mm256_storeu_ps(dst + x, res);
                }
#endif

                for (; x < image.cols; ++x)
                {
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
    int num_tasks)
{
    CV_Assert(image.type() == CV_32F);
    CV_Assert(image.cols >= 2);

    gx.create(image.size(), CV_32F);
    gy.create(image.size(), CV_32F);

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                const float* row = image.ptr<float>(y);
                const float* up  = image.ptr<float>(std::max(0, y - 1));
                const float* dn  = image.ptr<float>(std::min(image.rows - 1, y + 1));

                float* dx = gx.ptr<float>(y);
                float* dy = gy.ptr<float>(y);

                dx[0] = 0.5f * (row[1] - row[0]);

                int x = 1;

#if defined(__AVX2__)
                const __m256 half = _mm256_set1_ps(0.5f);

                for (; x + 7 < image.cols - 1; x += 8)
                {
                    const __m256 right = _mm256_loadu_ps(row + x + 1);
                    const __m256 left  = _mm256_loadu_ps(row + x - 1);
                    const __m256 ddx   = _mm256_mul_ps(_mm256_sub_ps(right, left), half);
                    _mm256_storeu_ps(dx + x, ddx);
                }
#endif

                for (; x < image.cols - 1; ++x)
                {
                    dx[x] = 0.5f * (row[x + 1] - row[x - 1]);
                }

                const int last = image.cols - 1;
                dx[last] = 0.5f * (row[last] - row[last - 1]);

                int xx = 0;

#if defined(__AVX2__)
                const __m256 halfy = _mm256_set1_ps(0.5f);
                for (; xx + 7 < image.cols; xx += 8)
                {
                    const __m256 upv = _mm256_loadu_ps(up + xx);
                    const __m256 dnv = _mm256_loadu_ps(dn + xx);
                    const __m256 ddy = _mm256_mul_ps(_mm256_sub_ps(dnv, upv), halfy);
                    _mm256_storeu_ps(dy + xx, ddy);
                }
#endif

                for (; xx < image.cols; ++xx)
                {
                    dy[xx] = 0.5f * (dn[xx] - up[xx]);
                }
            }
        });
}
