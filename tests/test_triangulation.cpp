#include "vio/stereo_triangulation.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

struct SyntheticScene {
    cv::Mat image1, image2;
    cv::Mat K;
    cv::Mat R;
    cv::Mat t;
    std::vector<cv::Point3d> points;
};

static void drawTexturedPatch(cv::Mat& img, cv::Point2d center, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> gray_dist(60, 240);
    int g1 = gray_dist(rng), g2 = gray_dist(rng), g3 = gray_dist(rng);

    cv::circle(img, center, 10, cv::Scalar(g1), -1);
    cv::circle(img, center, 14, cv::Scalar(g2), 2);
    cv::line(img, center - cv::Point2d(8, 8), center + cv::Point2d(8, 8), cv::Scalar(g3), 2);
    cv::line(img, center - cv::Point2d(8, -8), center + cv::Point2d(8, -8), cv::Scalar(g3), 2);
}

static SyntheticScene generateScene(int img_w, int img_h, int num_points, double baseline) {
    SyntheticScene scene;

    double fx = static_cast<double>(img_w);
    double fy = fx;
    double cx = img_w / 2.0;
    double cy = img_h / 2.0;
    scene.K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                          0, fy, cy,
                                          0,  0,  1);

    scene.R = cv::Mat::eye(3, 3, CV_64F);
    scene.t = (cv::Mat_<double>(3, 1) << baseline, 0, 0);

    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    scene.K.copyTo(P1(cv::Rect(0, 0, 3, 3)));

    cv::Mat Rt2;
    cv::hconcat(scene.R, scene.t, Rt2);
    cv::Mat P2 = scene.K * Rt2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> x_dist(-3.0, 3.0);
    std::uniform_real_distribution<double> y_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> z_dist(4.0, 12.0);

    // Uniform gray background (minimal noise to avoid spurious matches)
    scene.image1 = cv::Mat(img_h, img_w, CV_8UC1, cv::Scalar(120));
    scene.image2 = cv::Mat(img_h, img_w, CV_8UC1, cv::Scalar(120));

    for (int i = 0; i < num_points; ++i) {
        cv::Point3d pt(x_dist(rng), y_dist(rng), z_dist(rng));

        cv::Mat p4d = (cv::Mat_<double>(4, 1) << pt.x, pt.y, pt.z, 1.0);
        cv::Mat proj1 = P1 * p4d;
        cv::Mat proj2 = P2 * p4d;

        cv::Point2d uv1(proj1.at<double>(0) / proj1.at<double>(2),
                        proj1.at<double>(1) / proj1.at<double>(2));
        cv::Point2d uv2(proj2.at<double>(0) / proj2.at<double>(2),
                        proj2.at<double>(1) / proj2.at<double>(2));

        const int margin = 20;
        if (uv1.x < margin || uv1.x >= img_w - margin ||
            uv1.y < margin || uv1.y >= img_h - margin ||
            uv2.x < margin || uv2.x >= img_w - margin ||
            uv2.y < margin || uv2.y >= img_h - margin)
            continue;

        scene.points.push_back(pt);
        drawTexturedPatch(scene.image1, uv1, i * 7 + 13);
        drawTexturedPatch(scene.image2, uv2, i * 7 + 13);
    }

    return scene;
}

static double nearestDistance(const Eigen::Vector3d& reconstructed,
                              const std::vector<cv::Point3d>& gt_points) {
    double min_dist = 1e9;
    for (const auto& gt : gt_points) {
        Eigen::Vector3d g(gt.x, gt.y, gt.z);
        double d = (reconstructed - g).norm();
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

struct TestResult {
    std::string name;
    int gt_points = 0;
    int reconstructed = 0;
    double mean_error = -1;
    double median_error = -1;
    double max_error = -1;
    double pct_within_threshold = 0;
    bool exception = false;
    std::string exception_msg;
};

static TestResult runTest(const std::string& name, int img_w, int img_h,
                          int num_points, double baseline,
                          const vio::TriangulationConfig& cfg) {
    TestResult res;
    res.name = name;

    auto scene = generateScene(img_w, img_h, num_points, baseline);
    res.gt_points = static_cast<int>(scene.points.size());

    std::string path1 = "/tmp/vio_test_left.png";
    std::string path2 = "/tmp/vio_test_right.png";
    cv::imwrite(path1, scene.image1);
    cv::imwrite(path2, scene.image2);

    try {
        auto result = vio::triangulateFromImages(path1, path2, cfg);
        res.reconstructed = static_cast<int>(result.cloud.size());

        if (res.reconstructed == 0)
            return res;

        // Estimate scale (recoverPose returns unit-norm t)
        double gt_avg_dist = 0;
        for (const auto& p : scene.points)
            gt_avg_dist += std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        gt_avg_dist /= scene.points.size();

        double recon_avg_dist = 0;
        for (const auto& p : result.cloud)
            recon_avg_dist += p.position.norm();
        recon_avg_dist /= result.cloud.size();

        double scale = gt_avg_dist / recon_avg_dist;

        std::vector<double> errors;
        for (const auto& p : result.cloud) {
            Eigen::Vector3d scaled = p.position * scale;
            errors.push_back(nearestDistance(scaled, scene.points));
        }

        std::sort(errors.begin(), errors.end());
        res.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        res.median_error = errors[errors.size() / 2];
        res.max_error = errors.back();

        double threshold = gt_avg_dist * 0.05;
        int within = 0;
        for (double e : errors)
            if (e < threshold) ++within;
        res.pct_within_threshold = 100.0 * within / errors.size();

    } catch (const vio::TriangulationError& e) {
        res.exception = true;
        res.exception_msg = e.what();
    }

    return res;
}

static void printResult(const TestResult& r) {
    std::cout << "  " << r.name << "\n";
    if (r.exception) {
        std::cout << "    Exception: " << r.exception_msg << "\n";
    } else {
        std::cout << "    GT: " << r.gt_points
                  << "  Reconstructed: " << r.reconstructed << "\n";
        if (r.mean_error >= 0) {
            std::cout << "    Mean err:   " << r.mean_error
                      << "  Median err: " << r.median_error
                      << "  Max err: " << r.max_error << "\n";
            std::cout << "    Within 5% of depth: " << r.pct_within_threshold << "%\n";
        }
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== Stereo Triangulation Accuracy Tests ===\n\n";

    vio::TriangulationConfig cfg;
    cfg.show_matches = false;

    int pass_count = 0;
    int total = 0;

    // Test 1: Standard scene — should reconstruct well
    {
        auto r = runTest("Standard (640x480, 200pts, baseline=0.5)",
                         640, 480, 200, 0.5, cfg);
        bool ok = !r.exception && r.reconstructed > r.gt_points * 0.3
                  && r.median_error >= 0 && r.median_error < 0.5;
        std::cout << (ok ? "[PASS]" : "[FAIL]");
        printResult(r);
        if (ok) ++pass_count;
        ++total;
    }

    // Test 2: Wide baseline — harder matching, but should work
    {
        auto r = runTest("Wide baseline (640x480, 200pts, baseline=2.0)",
                         640, 480, 200, 2.0, cfg);
        bool ok = !r.exception && r.reconstructed > 20
                  && r.median_error >= 0 && r.median_error < 1.0;
        std::cout << (ok ? "[PASS]" : "[FAIL]");
        printResult(r);
        if (ok) ++pass_count;
        ++total;
    }

    // Test 3: Narrow baseline — expected to fail or have very few points
    // (degenerate geometry, this is a known limitation)
    {
        auto r = runTest("Narrow baseline (640x480, 200pts, baseline=0.05)",
                         640, 480, 200, 0.05, cfg);
        // Should either throw or reconstruct very few points (parallax filter kicks in)
        bool ok = r.exception || r.reconstructed < r.gt_points * 0.5;
        std::cout << (ok ? "[PASS]" : "[FAIL]");
        printResult(r);
        std::cout << "    (Expected: fails gracefully or few points due to degenerate geometry)\n\n";
        if (ok) ++pass_count;
        ++total;
    }

    // Test 4: High-res — more pixels, should be more accurate
    {
        auto r = runTest("High-res (1280x960, 300pts, baseline=0.5)",
                         1280, 960, 300, 0.5, cfg);
        bool ok = !r.exception && r.reconstructed > r.gt_points * 0.3
                  && r.median_error >= 0 && r.median_error < 0.5;
        std::cout << (ok ? "[PASS]" : "[FAIL]");
        printResult(r);
        if (ok) ++pass_count;
        ++total;
    }

    // Test 5: Moderate baseline with fewer points
    {
        auto r = runTest("Moderate (640x480, 80pts, baseline=0.3)",
                         640, 480, 80, 0.3, cfg);
        bool ok = !r.exception && r.reconstructed > 15
                  && r.median_error >= 0 && r.median_error < 0.8;
        std::cout << (ok ? "[PASS]" : "[FAIL]");
        printResult(r);
        if (ok) ++pass_count;
        ++total;
    }

    // Test 6: Error handling — missing file
    {
        ++total;
        bool ok = false;
        try {
            vio::triangulateFromImages("/tmp/nonexistent.png", "/tmp/also_nonexistent.png", cfg);
        } catch (const vio::TriangulationError&) {
            ok = true;
        }
        std::cout << (ok ? "[PASS]" : "[FAIL]")
                  << "  Error handling: missing files throw TriangulationError\n\n";
        if (ok) ++pass_count;
    }

    std::cout << "=== Summary: " << pass_count << " / " << total << " tests passed ===\n";

    return (pass_count == total) ? 0 : 1;
}
