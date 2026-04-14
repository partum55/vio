#include "gaussian_blur.hpp"
#include "tpool_default.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace
{
    double mean(const std::vector<double>& values)
    {
        if (values.empty()) return 0.0;
        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        return sum / static_cast<double>(values.size());
    }

    double stddev(const std::vector<double>& values, double avg)
    {
        if (values.size() < 2) return 0.0;

        double acc = 0.0;
        for (double v : values)
        {
            const double d = v - avg;
            acc += d * d;
        }
        return std::sqrt(acc / static_cast<double>(values.size() - 1));
    }

    double amdahlParallelFraction(double speedup, int threads)
    {
        if (threads <= 1 || speedup <= 0.0)
        {
            return 0.0;
        }

        const double denominator = 1.0 - (1.0 / static_cast<double>(threads));
        if (std::abs(denominator) < 1e-12)
        {
            return 0.0;
        }

        const double alpha = (1.0 - (1.0 / speedup)) / denominator;
        return std::clamp(alpha, 0.0, 1.0);
    }

    cv::Mat makeRandomImage(int rows, int cols, unsigned int seed)
    {
        cv::Mat image(rows, cols, CV_32F);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 255.0f);

        for (int y = 0; y < rows; ++y)
        {
            float* row = image.ptr<float>(y);
            for (int x = 0; x < cols; ++x)
            {
                row[x] = dist(rng);
            }
        }

        return image;
    }

    struct BenchmarkResult
    {
        int threads = 1;
        int tasks = 1;
        double mean_ms = 0.0;
        double std_ms = 0.0;
        double speedup = 1.0;
        double efficiency = 1.0;
        double parallel_fraction = 0.0;
        double checksum = 0.0;
    };
}

int main(int argc, char** argv)
{
    int rows = 2160;
    int cols = 3840;
    int repeats = 50;
    int warmup = 10;
    int max_threads = 0;
    int tasks_per_thread = 4;
    std::string csv_path = "convolution_benchmark.csv";

    if (argc >= 2) rows = std::stoi(argv[1]);
    if (argc >= 3) cols = std::stoi(argv[2]);
    if (argc >= 4) repeats = std::stoi(argv[3]);
    if (argc >= 5) warmup = std::stoi(argv[4]);
    if (argc >= 6) max_threads = std::stoi(argv[5]);
    if (argc >= 7) tasks_per_thread = std::stoi(argv[6]);
    if (argc >= 8) csv_path = argv[7];

    if (rows < 2 || cols < 2)
    {
        std::cerr << "rows and cols must be >= 2\n";
        return 1;
    }

    if (repeats <= 0 || warmup < 0 || tasks_per_thread <= 0)
    {
        std::cerr << "repeats must be > 0, warmup >= 0, tasks_per_thread > 0\n";
        return 1;
    }

    const unsigned int hw = std::thread::hardware_concurrency();
    if (max_threads <= 0)
    {
        max_threads = (hw == 0) ? 8 : static_cast<int>(hw);
    }

    const cv::Mat image = makeRandomImage(rows, cols, 42);

    std::vector<BenchmarkResult> results;
    results.reserve(max_threads);

    double baseline_ms = std::numeric_limits<double>::quiet_NaN();
    volatile double checksum_accumulator = 0.0;

    for (int threads = 1; threads <= max_threads; ++threads)
    {
        ThreadPool pool(threads);
        ABCThreadPool& abstract_pool = pool;
        const int num_tasks = std::max(1, threads * tasks_per_thread);

        cv::Mat result;

        for (int i = 0; i < warmup; ++i)
        {
            gaussianBlurCustomBanded(image, result, abstract_pool, threads, 64);
            checksum_accumulator += result.at<float>(rows / 2, cols / 2);
        }

        std::vector<double> times_ms;
        times_ms.reserve(repeats);
        double checksum = 0.0;

        for (int i = 0; i < repeats; ++i)
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            gaussianBlurCustomBanded(image, result, abstract_pool, threads, 64);
            const auto t1 = std::chrono::high_resolution_clock::now();

            const std::chrono::duration<double, std::milli> dt = t1 - t0;
            times_ms.push_back(dt.count());

            checksum += result.at<float>(i % rows, i % cols);
        }

        const double avg = mean(times_ms);
        const double sd = stddev(times_ms, avg);

        if (threads == 1)
        {
            baseline_ms = avg;
        }

        const double speedup = baseline_ms / avg;
        const double efficiency = speedup / static_cast<double>(threads);
        const double parallel_fraction = amdahlParallelFraction(speedup, threads);

        results.push_back(BenchmarkResult{
            threads,
            num_tasks,
            avg,
            sd,
            speedup,
            efficiency,
            parallel_fraction,
            checksum
        });

        std::cout << "threads=" << threads
                  << " tasks=" << num_tasks
                  << " mean_ms=" << avg
                  << " std_ms=" << sd
                  << " speedup=" << speedup
                  << " efficiency=" << efficiency
                  << " parallel_fraction=" << parallel_fraction
                  << " checksum=" << checksum
                  << "\n";
    }

    std::ofstream csv(csv_path);
    if (!csv)
    {
        std::cerr << "Failed to open output CSV: " << csv_path << "\n";
        return 1;
    }

    csv << "threads,tasks,mean_ms,std_ms,speedup,efficiency,parallel_fraction,rows,cols,repeats,warmup\n";
    csv << std::fixed << std::setprecision(6);

    for (const auto& r : results)
    {
        csv << r.threads << ','
            << r.tasks << ','
            << r.mean_ms << ','
            << r.std_ms << ','
            << r.speedup << ','
            << r.efficiency << ','
            << r.parallel_fraction << ','
            << rows << ','
            << cols << ','
            << repeats << ','
            << warmup << '\n';
    }

    std::cout << "Saved CSV to: " << csv_path << "\n";
    std::cout << "Ignore checksum accumulator: " << checksum_accumulator << "\n";
    return 0;
}
