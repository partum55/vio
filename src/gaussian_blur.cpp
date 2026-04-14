#include "gaussian_blur.hpp"
#include "parallel_utils.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace
{
    inline void ensureFloatMat(cv::Mat& mat, const cv::Size& size)
    {
        mat.create(size, CV_32F);
    }

    inline void convolveHorizontalSingleRow(
        const float* src,
        float* dst,
        int cols,
        const float kernel[3])
    {
        dst[0] = kernel[0] * src[0] +
                 kernel[1] * src[0] +
                 kernel[2] * src[1];

        int x = 1;

#if defined(__AVX2__)
        const __m256 k0 = _mm256_set1_ps(kernel[0]);
        const __m256 k1 = _mm256_set1_ps(kernel[1]);
        const __m256 k2 = _mm256_set1_ps(kernel[2]);

        for (; x + 7 < cols - 1; x += 8)
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

        for (; x < cols - 1; ++x)
        {
            dst[x] = kernel[0] * src[x - 1] +
                     kernel[1] * src[x] +
                     kernel[2] * src[x + 1];
        }

        const int last = cols - 1;
        dst[last] = kernel[0] * src[last - 1] +
                    kernel[1] * src[last] +
                    kernel[2] * src[last];
    }

    inline void convolveVerticalThreeRows(
        const float* row_up,
        const float* row_mid,
        const float* row_down,
        float* dst,
        int cols,
        const float kernel[3])
    {
        int x = 0;

#if defined(__AVX2__)
        const __m256 k0 = _mm256_set1_ps(kernel[0]);
        const __m256 k1 = _mm256_set1_ps(kernel[1]);
        const __m256 k2 = _mm256_set1_ps(kernel[2]);

        for (; x + 7 < cols; x += 8)
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

        for (; x < cols; ++x)
        {
            dst[x] = kernel[0] * row_up[x] +
                     kernel[1] * row_mid[x] +
                     kernel[2] * row_down[x];
        }
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

    parallel_for_rows(
        pool,
        image.rows,
        num_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                const float* src = image.ptr<float>(y);
                float* dst = result.ptr<float>(y);
                convolveHorizontalSingleRow(src, dst, image.cols, kernel);
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

    parallel_for_rows(
        pool,
        image.rows,
        num_tasks,
        [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            {
                float* dst = result.ptr<float>(y);

                const float* row_up   = image.ptr<float>(std::max(0, y - 1));
                const float* row_mid  = image.ptr<float>(y);
                const float* row_down = image.ptr<float>(std::min(image.rows - 1, y + 1));

                convolveVerticalThreeRows(
                    row_up,
                    row_mid,
                    row_down,
                    dst,
                    image.cols,
                    kernel);
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

void gaussianBlurCustomBanded(
    const cv::Mat& image,
    cv::Mat& result,
    ABCThreadPool& pool,
    int num_threads,
    int rows_per_chunk)
{
    CV_Assert(image.type() == CV_32F);
    CV_Assert(image.cols >= 2);

    constexpr float kernel[3] = {0.25f, 0.5f, 0.25f};

    ensureFloatMat(result, image.size());

    const int threads = std::max(1, num_threads);
    const int chunk_rows = std::max(8, rows_per_chunk);

    std::atomic<int> next_row{0};
    std::vector<std::future<void>> futures;
    futures.reserve(threads);

    for (int t = 0; t < threads; ++t)
    {
        futures.emplace_back(
            pool.submit_task(
                [&, chunk_rows]()
                {
                    cv::Mat localTemp(chunk_rows + 2, image.cols, CV_32F);

                    while (true)
                    {
                        const int y0 = next_row.fetch_add(chunk_rows);
                        if (y0 >= image.rows)
                        {
                            break;
                        }

                        const int y1 = std::min(image.rows, y0 + chunk_rows);
                        const int local_rows = y1 - y0;

                        for (int ly = 0; ly < local_rows + 2; ++ly)
                        {
                            const int gy = std::clamp(y0 + ly - 1, 0, image.rows - 1);
                            const float* src = image.ptr<float>(gy);
                            float* dst = localTemp.ptr<float>(ly);

                            convolveHorizontalSingleRow(src, dst, image.cols, kernel);
                        }

                        for (int y = y0; y < y1; ++y)
                        {
                            const int ly = (y - y0) + 1;

                            const float* up   = localTemp.ptr<float>(ly - 1);
                            const float* mid  = localTemp.ptr<float>(ly);
                            const float* down = localTemp.ptr<float>(ly + 1);
                            float* dst = result.ptr<float>(y);

                            convolveVerticalThreeRows(
                                up,
                                mid,
                                down,
                                dst,
                                image.cols,
                                kernel);
                        }
                    }
                }));
    }

    for (auto& f : futures)
    {
        f.get();
    }
}

cv::Mat gaussianBlurCustomBanded(
    const cv::Mat& image,
    ABCThreadPool& pool,
    int num_threads,
    int rows_per_chunk)
{
    cv::Mat result;
    gaussianBlurCustomBanded(image, result, pool, num_threads, rows_per_chunk);
    return result;
}
