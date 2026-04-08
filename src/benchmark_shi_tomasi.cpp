#include "shi_tomasi.hpp"
#include "tpool_default.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <numeric>
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
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr
            << "Usage: benchmark_shi_tomasi <image_path> [num_threads] [repeats] [warmup]\n"
            << "Example: ./benchmark_shi_tomasi image.png 8 50 5\n";
        return 1;
    }

    const std::string image_path = argv[1];

    int num_threads = 0;
    if (argc >= 3)
    {
        num_threads = std::stoi(argv[2]);
    }
    else
    {
        const unsigned int hw = std::thread::hardware_concurrency();
        num_threads = (hw == 0) ? 4 : static_cast<int>(hw);
    }

    int repeats = 30;
    if (argc >= 4)
    {
        repeats = std::stoi(argv[3]);
    }

    int warmup = 5;
    if (argc >= 5)
    {
        warmup = std::stoi(argv[4]);
    }

    if (num_threads <= 0)
    {
        std::cerr << "Invalid num_threads: " << num_threads << "\n";
        return 1;
    }

    if (repeats <= 0)
    {
        std::cerr << "Invalid repeats: " << repeats << "\n";
        return 1;
    }

    if (warmup < 0)
    {
        std::cerr << "Invalid warmup: " << warmup << "\n";
        return 1;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Failed to read image: " << image_path << "\n";
        return 1;
    }

    ThreadPool pool(num_threads);
    ABCThreadPool& abstract_pool = pool;

    ShiTomasiParams p;
    p.maxCorners = 500;
    p.qualityLevel = 0.01;
    p.minDistance = 10.0;
    p.blockSize = 5;
    p.gaussianSigma = 1.0;
    p.nmsRadius = 2;

    CustomShiTomasiDetector detector(abstract_pool, num_threads);

    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        volatile auto pts = detector.detect(img, p);
        (void)pts;
    }

    std::vector<double> times_ms;
    times_ms.reserve(repeats);

    std::size_t total_points = 0;

    for (int i = 0; i < repeats; ++i)
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto pts = detector.detect(img, p);
        const auto t1 = std::chrono::high_resolution_clock::now();

        const std::chrono::duration<double, std::milli> dt = t1 - t0;
        times_ms.push_back(dt.count());
        total_points += pts.size();
    }

    const double avg = mean(times_ms);
    const double sd = stddev(times_ms, avg);

    std::cout << "threads=" << num_threads
              << " repeats=" << repeats
              << " warmup=" << warmup
              << " mean_ms=" << avg
              << " std_ms=" << sd
              << " avg_points=" << static_cast<double>(total_points) / repeats
              << "\n";

    return 0;
}
