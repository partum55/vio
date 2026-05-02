#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace vio {

class VisionComputeBackend {
public:
    virtual ~VisionComputeBackend() = default;

    virtual const std::string& name() const = 0;
    virtual bool isGpu() const = 0;

    virtual void trackPyramidalLK(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const std::vector<cv::Point2f>& pts0,
        std::vector<cv::Point2f>& pts1,
        std::vector<uchar>& status,
        std::vector<float>& err,
        int win_size,
        int max_level,
        int max_iters,
        float eps
    ) const = 0;

    virtual void trackPyramidalLKWithGuess(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const std::vector<cv::Point2f>& pts0,
        const std::vector<cv::Point2f>& initial_guess,
        std::vector<cv::Point2f>& pts1,
        std::vector<uchar>& status,
        std::vector<float>& err,
        int win_size,
        int max_level,
        int max_iters,
        float eps
    ) const = 0;

    // src must be CV_32F; dst receives same-size CV_32F blurred result
    virtual void gaussianBlur(const cv::Mat& src, cv::Mat& dst) const = 0;

    // src must be CV_32F; gx/gy receive same-size CV_32F central-difference gradients
    virtual void sobelGradients(const cv::Mat& src, cv::Mat& gx, cv::Mat& gy) const = 0;

    static std::shared_ptr<VisionComputeBackend> createAuto();
};

} // namespace vio
