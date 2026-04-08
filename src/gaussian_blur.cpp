#include "gaussian_blur.hpp"
#include "parallel_utils.hpp"

#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace
{
    inline void ensureFloatMat(cv::Mat& mat, const cv::Size& size)
    {
        mat.create(size, CV_32F);
    }
}

void convolveHorizontalReplicate(
    const cv::Mat& image,
    cv::Mat& result,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks)
{
    CV_Assert(image.type() == CV_32F);
    CV_Assert(image.cols >= 2);

    ensureFloatMat(result, image.size());

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                const float* src = image.ptr<float>(y);
                float* dst = result.ptr<float>(y);

                dst[0] = kernel[0] * src[0] +
                         kernel[1] * src[0] +
                         kernel[2] * src[1];

                int x = 1;

#if defined(__AVX2__)
                const __m256 k0 = _mm256_set1_ps(kernel[0]);
                const __m256 k1 = _mm256_set1_ps(kernel[1]);
                const __m256 k2 = _mm256_set1_ps(kernel[2]);

                for (; x + 7 < image.cols - 1; x += 8)
                {
                    const __m256 left  = _mm256_loadu_ps(src + x - 1);
                    const __m256 mid   = _mm256_loadu_ps(src + x);
                    const __m256 right = _mm256_loadu_ps(src + x + 1);

                    __m256 sum = _mm256_mul_ps(left, k0);
                    sum = _mm256_fmadd_ps(mid, k1, sum);
                    sum = _mm256_fmadd_ps(right, k2, sum);

                    _mm256_storeu_ps(dst + x, sum);
                }
#endif

                for (; x < image.cols - 1; ++x)
                {
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
}

void convolveVerticalReplicate(
    const cv::Mat& image,
    cv::Mat& result,
    const float kernel[3],
    ABCThreadPool& pool,
    int num_tasks)
{
    CV_Assert(image.type() == CV_32F);

    ensureFloatMat(result, image.size());

    parallel_for_rows(pool, image.rows, num_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                float* dst = result.ptr<float>(y);

                const float* row_up   = image.ptr<float>(std::max(0, y - 1));
                const float* row_mid  = image.ptr<float>(y);
                const float* row_down = image.ptr<float>(std::min(image.rows - 1, y + 1));

                int x = 0;

#if defined(__AVX2__)
                const __m256 k0 = _mm256_set1_ps(kernel[0]);
                const __m256 k1 = _mm256_set1_ps(kernel[1]);
                const __m256 k2 = _mm256_set1_ps(kernel[2]);

                for (; x + 7 < image.cols; x += 8)
                {
                    const __m256 up   = _mm256_loadu_ps(row_up + x);
                    const __m256 mid  = _mm256_loadu_ps(row_mid + x);
                    const __m256 down = _mm256_loadu_ps(row_down + x);

                    __m256 sum = _mm256_mul_ps(up, k0);
                    sum = _mm256_fmadd_ps(mid, k1, sum);
                    sum = _mm256_fmadd_ps(down, k2, sum);

                    _mm256_storeu_ps(dst + x, sum);
                }
#endif

                for (; x < image.cols; ++x)
                {
                    dst[x] = kernel[0] * row_up[x] +
                             kernel[1] * row_mid[x] +
                             kernel[2] * row_down[x];
                }
            }
        });
}

void gaussianBlurCustom(
    const cv::Mat& image,
    cv::Mat& result,
    ABCThreadPool& pool,
    int num_tasks)
{
    CV_Assert(image.type() == CV_32F);

    constexpr float kernel[3] = {0.25f, 0.5f, 0.25f};

    cv::Mat temp;
    temp.create(image.size(), CV_32F);

    convolveHorizontalReplicate(image, temp, kernel, pool, num_tasks);
    convolveVerticalReplicate(temp, result, kernel, pool, num_tasks);
}

cv::Mat gaussianBlurCustom(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_tasks)
{
    cv::Mat result;
    gaussianBlurCustom(image, result, pool, num_tasks);
    return result;
}
