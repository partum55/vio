#include "frontend/vision_compute_backend.hpp"

#include "keypoints/gaussian_blur.hpp"
#include "keypoints/sobel.hpp"
#include "keypoints/tpool_default.hpp"
#include "tracking/lk_tracker.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(VIO_HAVE_OPENCL)
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

namespace {

std::string gpuModeFromEnv()
{
    const char* raw = std::getenv("VIO_GPU");
    std::string mode = raw == nullptr ? "auto" : raw;
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode != "auto" && mode != "on" && mode != "off") {
        std::cerr << "Unknown VIO_GPU='" << mode << "', using auto\n";
        return "auto";
    }
    return mode;
}

class CpuVisionComputeBackend final : public VisionComputeBackend {
public:
    explicit CpuVisionComputeBackend(std::string reason = {})
        : name_(reason.empty() ? "CPU custom LK" : "CPU custom LK (" + reason + ")")
        , num_threads_(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())))
        , pool_(num_threads_)
    {
    }

    const std::string& name() const override
    {
        return name_;
    }

    bool isGpu() const override
    {
        return false;
    }

    void gaussianBlur(const cv::Mat& src, cv::Mat& dst) const override
    {
        dst = vio::gaussianBlurCustom(src, pool_, num_threads_);
    }

    void sobelGradients(const cv::Mat& src, cv::Mat& gx, cv::Mat& gy) const override
    {
        vio::centralDifferenceXY(src, gx, gy, pool_, num_threads_);
    }

    void trackPyramidalLK(
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
    ) const override {
        trackPointsPyramidalLK(
            prev_gray,
            curr_gray,
            pts0,
            pts1,
            status,
            err,
            win_size,
            max_level,
            max_iters,
            eps
        );
    }

    void trackPyramidalLKWithGuess(
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
    ) const override {
        trackPointsPyramidalLKWithGuess(
            prev_gray,
            curr_gray,
            pts0,
            initial_guess,
            pts1,
            status,
            err,
            win_size,
            max_level,
            max_iters,
            eps
        );
    }

private:
    std::string name_;
    int num_threads_;
    mutable vio::ThreadPool pool_;
};

#if defined(VIO_HAVE_OPENCL)

const char* kOpenClSource = R"CLC(
inline int clampi(int v, int lo, int hi)
{
    return min(max(v, lo), hi);
}

inline float bilinear(__global const float* img, int rows, int cols, float x, float y)
{
    if (x < 0.0f || y < 0.0f || x >= (float)(cols - 1) || y >= (float)(rows - 1)) {
        return 0.0f;
    }

    const int x0 = (int)x;
    const int y0 = (int)y;
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float a = x - (float)x0;
    const float b = y - (float)y0;

    const float i00 = img[y0 * cols + x0];
    const float i10 = img[y0 * cols + x1];
    const float i01 = img[y1 * cols + x0];
    const float i11 = img[y1 * cols + x1];

    return (1.0f - a) * (1.0f - b) * i00 +
           a * (1.0f - b) * i10 +
           (1.0f - a) * b * i01 +
           a * b * i11;
}

__kernel void u8_to_float(__global const uchar* src, __global float* dst, int count)
{
    const int i = get_global_id(0);
    if (i >= count) {
        return;
    }
    dst[i] = (float)src[i] * (1.0f / 255.0f);
}

__kernel void blur_downsample(
    __global const float* src,
    __global float* dst,
    int src_rows,
    int src_cols,
    int dst_rows,
    int dst_cols
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= dst_cols || y >= dst_rows) {
        return;
    }

    const int cx = min(2 * x, src_cols - 1);
    const int cy = min(2 * y, src_rows - 1);

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        const int yy = clampi(cy + ky, 0, src_rows - 1);
        for (int kx = -1; kx <= 1; ++kx) {
            const int xx = clampi(cx + kx, 0, src_cols - 1);
            const int ax = kx == 0 ? 2 : 1;
            const int ay = ky == 0 ? 2 : 1;
            sum += src[yy * src_cols + xx] * (float)(ax * ay);
        }
    }
    dst[y * dst_cols + x] = sum * (1.0f / 16.0f);
}

__kernel void gaussian_blur(
    __global const float* src,
    __global float* dst,
    int rows,
    int cols
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= cols || y >= rows) {
        return;
    }

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        const int yy = clampi(y + ky, 0, rows - 1);
        for (int kx = -1; kx <= 1; ++kx) {
            const int xx = clampi(x + kx, 0, cols - 1);
            const int ax = kx == 0 ? 2 : 1;
            const int ay = ky == 0 ? 2 : 1;
            sum += src[yy * cols + xx] * (float)(ax * ay);
        }
    }
    dst[y * cols + x] = sum * (1.0f / 16.0f);
}

__kernel void sobel_gradients(
    __global const float* img,
    __global float* ix,
    __global float* iy,
    int rows,
    int cols
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= cols || y >= rows) {
        return;
    }

    const int xm1 = clampi(x - 1, 0, cols - 1);
    const int xp1 = clampi(x + 1, 0, cols - 1);
    const int ym1 = clampi(y - 1, 0, rows - 1);
    const int yp1 = clampi(y + 1, 0, rows - 1);

    const float a00 = img[ym1 * cols + xm1];
    const float a01 = img[ym1 * cols + x];
    const float a02 = img[ym1 * cols + xp1];
    const float a10 = img[y * cols + xm1];
    const float a12 = img[y * cols + xp1];
    const float a20 = img[yp1 * cols + xm1];
    const float a21 = img[yp1 * cols + x];
    const float a22 = img[yp1 * cols + xp1];

    ix[y * cols + x] =
        -a00 + a02 -
        2.0f * a10 + 2.0f * a12 -
        a20 + a22;
    iy[y * cols + x] =
        -a00 - 2.0f * a01 - a02 +
         a20 + 2.0f * a21 + a22;
}

__kernel void init_flow_zero(
    __global float2* flow,
    __global uchar* status,
    __global float* err,
    int count
) {
    const int i = get_global_id(0);
    if (i >= count) {
        return;
    }
    flow[i] = (float2)(0.0f, 0.0f);
    status[i] = 1;
    err[i] = -1.0f;
}

__kernel void init_flow_guess(
    __global const float2* pts0,
    __global const float2* guess,
    __global float2* flow,
    __global uchar* status,
    __global float* err,
    float inv_top_scale,
    int count
) {
    const int i = get_global_id(0);
    if (i >= count) {
        return;
    }
    flow[i] = (guess[i] - pts0[i]) * inv_top_scale;
    status[i] = 1;
    err[i] = -1.0f;
}

__kernel void track_level(
    __global const float* prev,
    __global const float* curr,
    __global const float* ix,
    __global const float* iy,
    __global const float2* pts0,
    __global float2* flow,
    __global uchar* status,
    __global float* err,
    int rows,
    int cols,
    float level_scale,
    int win_size,
    int max_iters,
    float eps,
    int count
) {
    const int i = get_global_id(0);
    if (i >= count || status[i] == 0) {
        return;
    }

    const int r = win_size / 2;
    const float2 pt_prev = pts0[i] * level_scale;
    float2 d = flow[i];

    if (pt_prev.x - (float)r < 0.0f || pt_prev.y - (float)r < 0.0f ||
        pt_prev.x + (float)r >= (float)(cols - 1) ||
        pt_prev.y + (float)r >= (float)(rows - 1)) {
        status[i] = 0;
        err[i] = -1.0f;
        return;
    }

    float sse = 0.0f;
    for (int iter = 0; iter < max_iters; ++iter) {
        const float cx = pt_prev.x + d.x;
        const float cy = pt_prev.y + d.y;

        if (cx - (float)r < 0.0f || cy - (float)r < 0.0f ||
            cx + (float)r >= (float)(cols - 1) ||
            cy + (float)r >= (float)(rows - 1)) {
            status[i] = 0;
            err[i] = -1.0f;
            return;
        }

        float h00 = 0.0f;
        float h01 = 0.0f;
        float h11 = 0.0f;
        float b0 = 0.0f;
        float b1 = 0.0f;
        sse = 0.0f;

        for (int py = -r; py <= r; ++py) {
            for (int px = -r; px <= r; ++px) {
                const float tx = pt_prev.x + (float)px;
                const float ty = pt_prev.y + (float)py;
                const float x = cx + (float)px;
                const float y = cy + (float)py;

                const float t = bilinear(prev, rows, cols, tx, ty);
                const float image = bilinear(curr, rows, cols, x, y);
                const float gx = bilinear(ix, rows, cols, x, y);
                const float gy = bilinear(iy, rows, cols, x, y);
                const float residual = image - t;

                h00 += gx * gx;
                h01 += gx * gy;
                h11 += gy * gy;
                b0 += gx * residual;
                b1 += gy * residual;
                sse += residual * residual;
            }
        }

        const float det = h00 * h11 - h01 * h01;
        if (fabs(det) < 1e-6f) {
            status[i] = 0;
            err[i] = -1.0f;
            return;
        }

        const float inv_det = 1.0f / det;
        const float delta_x = (-h11 * b0 + h01 * b1) * inv_det;
        const float delta_y = ( h01 * b0 - h00 * b1) * inv_det;
        d.x += delta_x;
        d.y += delta_y;

        if (delta_x * delta_x + delta_y * delta_y < eps * eps) {
            break;
        }
    }

    flow[i] = d;
    err[i] = sse / (float)(win_size * win_size);
}

__kernel void scale_flow(__global float2* flow, __global uchar* status, int count)
{
    const int i = get_global_id(0);
    if (i >= count || status[i] == 0) {
        return;
    }
    flow[i] *= 2.0f;
}

__kernel void finish_tracks(
    __global const float2* pts0,
    __global const float2* flow,
    __global float2* pts1,
    __global const uchar* status,
    int count
) {
    const int i = get_global_id(0);
    if (i >= count) {
        return;
    }
    pts1[i] = pts0[i] + flow[i];
    if (status[i] == 0) {
        pts1[i] = pts0[i] + flow[i];
    }
}
)CLC";

[[noreturn]] void throwCl(const std::string& where, cl_int err)
{
    throw std::runtime_error(where + " failed with OpenCL error " + std::to_string(err));
}

void checkCl(cl_int err, const std::string& where)
{
    if (err != CL_SUCCESS) {
        throwCl(where, err);
    }
}

std::string getDeviceString(cl_device_id device, cl_device_info key)
{
    size_t size = 0;
    cl_int err = clGetDeviceInfo(device, key, 0, nullptr, &size);
    if (err != CL_SUCCESS || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    err = clGetDeviceInfo(device, key, size, value.data(), nullptr);
    if (err != CL_SUCCESS) {
        return {};
    }
    while (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

struct ClBuffer {
    ClBuffer() = default;

    ClBuffer(cl_context context, size_t bytes, cl_mem_flags flags, const void* host_ptr = nullptr)
    {
        cl_int err = CL_SUCCESS;
        mem = clCreateBuffer(context, flags, bytes, const_cast<void*>(host_ptr), &err);
        checkCl(err, "clCreateBuffer");
    }

    ClBuffer(const ClBuffer&) = delete;
    ClBuffer& operator=(const ClBuffer&) = delete;

    ClBuffer(ClBuffer&& other) noexcept
        : mem(std::exchange(other.mem, nullptr))
    {
    }

    ClBuffer& operator=(ClBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            mem = std::exchange(other.mem, nullptr);
        }
        return *this;
    }

    ~ClBuffer()
    {
        release();
    }

    void release()
    {
        if (mem != nullptr) {
            clReleaseMemObject(mem);
            mem = nullptr;
        }
    }

    cl_mem mem = nullptr;
};

struct ClImage {
    int rows = 0;
    int cols = 0;
    ClBuffer data;
};

class OpenClVisionComputeBackend final : public VisionComputeBackend {
public:
    explicit OpenClVisionComputeBackend(bool strict)
        : strict_(strict)
    {
        initialize();
    }

    ~OpenClVisionComputeBackend() override
    {
        if (program_ != nullptr) {
            clReleaseProgram(program_);
        }
        if (queue_ != nullptr) {
            clReleaseCommandQueue(queue_);
        }
        if (context_ != nullptr) {
            clReleaseContext(context_);
        }
    }

    const std::string& name() const override
    {
        return name_;
    }

    bool isGpu() const override
    {
        return true;
    }

    void trackPyramidalLK(
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
    ) const override {
        try {
            runOpenCl(prev_gray, curr_gray, pts0, nullptr, pts1, status, err, win_size, max_level, max_iters, eps);
        } catch (const std::exception& ex) {
            handleRuntimeFailure(ex);
            cpu_fallback_.trackPyramidalLK(
                prev_gray, curr_gray, pts0, pts1, status, err, win_size, max_level, max_iters, eps);
        }
    }

    void trackPyramidalLKWithGuess(
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
    ) const override {
        try {
            runOpenCl(prev_gray, curr_gray, pts0, &initial_guess, pts1, status, err, win_size, max_level, max_iters, eps);
        } catch (const std::exception& ex) {
            handleRuntimeFailure(ex);
            cpu_fallback_.trackPyramidalLKWithGuess(
                prev_gray, curr_gray, pts0, initial_guess, pts1, status, err, win_size, max_level, max_iters, eps);
        }
    }

    void gaussianBlur(const cv::Mat& src_raw, cv::Mat& dst) const override
    {
        try {
            cv::Mat src = src_raw.isContinuous() ? src_raw : src_raw.clone();
            const size_t bytes = static_cast<size_t>(src.rows) * static_cast<size_t>(src.cols) * sizeof(float);
            ClBuffer src_buf = makeBuffer(bytes, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src.data);
            ClBuffer dst_buf = makeBuffer(bytes, CL_MEM_WRITE_ONLY);
            cl_mem src_mem = src_buf.mem;
            cl_mem dst_mem = dst_buf.mem;
            run2D("gaussian_blur", static_cast<size_t>(src.cols), static_cast<size_t>(src.rows), {
                {sizeof(cl_mem), &src_mem},
                {sizeof(cl_mem), &dst_mem},
                {sizeof(int), &src.rows},
                {sizeof(int), &src.cols},
            });
            dst.create(src.rows, src.cols, CV_32F);
            checkCl(clEnqueueReadBuffer(queue_, dst_buf.mem, CL_TRUE, 0, bytes, dst.data, 0, nullptr, nullptr),
                    "clEnqueueReadBuffer(gaussian_blur)");
        } catch (const std::exception& ex) {
            handleRuntimeFailure(ex);
            cpu_fallback_.gaussianBlur(src_raw, dst);
        }
    }

    void sobelGradients(const cv::Mat& src_raw, cv::Mat& gx, cv::Mat& gy) const override
    {
        try {
            cv::Mat src = src_raw.isContinuous() ? src_raw : src_raw.clone();
            const size_t bytes = static_cast<size_t>(src.rows) * static_cast<size_t>(src.cols) * sizeof(float);
            ClBuffer src_buf = makeBuffer(bytes, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src.data);
            ClBuffer gx_buf  = makeBuffer(bytes, CL_MEM_WRITE_ONLY);
            ClBuffer gy_buf  = makeBuffer(bytes, CL_MEM_WRITE_ONLY);
            cl_mem src_mem = src_buf.mem;
            cl_mem gx_mem  = gx_buf.mem;
            cl_mem gy_mem  = gy_buf.mem;
            run2D("sobel_gradients", static_cast<size_t>(src.cols), static_cast<size_t>(src.rows), {
                {sizeof(cl_mem), &src_mem},
                {sizeof(cl_mem), &gx_mem},
                {sizeof(cl_mem), &gy_mem},
                {sizeof(int), &src.rows},
                {sizeof(int), &src.cols},
            });
            gx.create(src.rows, src.cols, CV_32F);
            gy.create(src.rows, src.cols, CV_32F);
            checkCl(clEnqueueReadBuffer(queue_, gx_buf.mem, CL_TRUE, 0, bytes, gx.data, 0, nullptr, nullptr),
                    "clEnqueueReadBuffer(sobel_gx)");
            checkCl(clEnqueueReadBuffer(queue_, gy_buf.mem, CL_TRUE, 0, bytes, gy.data, 0, nullptr, nullptr),
                    "clEnqueueReadBuffer(sobel_gy)");
        } catch (const std::exception& ex) {
            handleRuntimeFailure(ex);
            cpu_fallback_.sobelGradients(src_raw, gx, gy);
        }
    }

private:
    void initialize()
    {
        cl_uint platform_count = 0;
        checkCl(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
        if (platform_count == 0) {
            throw std::runtime_error("no OpenCL platforms found");
        }

        std::vector<cl_platform_id> platforms(platform_count);
        checkCl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

        cl_int last_err = CL_SUCCESS;
        for (cl_platform_id platform : platforms) {
            cl_uint device_count = 0;
            last_err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
            if (last_err != CL_SUCCESS || device_count == 0) {
                continue;
            }

            std::vector<cl_device_id> devices(device_count);
            checkCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr), "clGetDeviceIDs");
            device_ = devices.front();

            cl_int err = CL_SUCCESS;
            context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
            checkCl(err, "clCreateContext");
            queue_ = clCreateCommandQueue(context_, device_, 0, &err);
            checkCl(err, "clCreateCommandQueue");

            const char* source = kOpenClSource;
            const size_t length = std::strlen(kOpenClSource);
            program_ = clCreateProgramWithSource(context_, 1, &source, &length, &err);
            checkCl(err, "clCreateProgramWithSource");

            err = clBuildProgram(program_, 1, &device_, "", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                size_t log_size = 0;
                clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::string log(log_size, '\0');
                if (log_size > 0) {
                    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                }
                throw std::runtime_error("OpenCL build failed: " + log);
            }

            name_ = "OpenCL custom LK - " + getDeviceString(device_, CL_DEVICE_NAME);
            return;
        }

        if (last_err != CL_SUCCESS) {
            throwCl("clGetDeviceIDs(CL_DEVICE_TYPE_GPU)", last_err);
        }
        throw std::runtime_error("no OpenCL GPU devices found");
    }

    ClBuffer makeBuffer(size_t bytes, cl_mem_flags flags, const void* host_ptr = nullptr) const
    {
        return ClBuffer(context_, bytes, flags, host_ptr);
    }

    cl_kernel makeKernel(const char* name) const
    {
        cl_int err = CL_SUCCESS;
        cl_kernel kernel = clCreateKernel(program_, name, &err);
        checkCl(err, std::string("clCreateKernel(") + name + ")");
        return kernel;
    }

    void run1D(const char* name, size_t global, const std::vector<std::pair<size_t, const void*>>& args) const
    {
        cl_kernel kernel = makeKernel(name);
        for (cl_uint i = 0; i < args.size(); ++i) {
            checkCl(clSetKernelArg(kernel, i, args[i].first, args[i].second), std::string("clSetKernelArg(") + name + ")");
        }
        const cl_int err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        clReleaseKernel(kernel);
        checkCl(err, std::string("clEnqueueNDRangeKernel(") + name + ")");
    }

    void run2D(const char* name, size_t cols, size_t rows, const std::vector<std::pair<size_t, const void*>>& args) const
    {
        cl_kernel kernel = makeKernel(name);
        for (cl_uint i = 0; i < args.size(); ++i) {
            checkCl(clSetKernelArg(kernel, i, args[i].first, args[i].second), std::string("clSetKernelArg(") + name + ")");
        }
        const size_t global[2] = {cols, rows};
        const cl_int err = clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        clReleaseKernel(kernel);
        checkCl(err, std::string("clEnqueueNDRangeKernel(") + name + ")");
    }

    static cv::Mat continuousGray(const cv::Mat& gray)
    {
        if (gray.empty()) {
            throw std::runtime_error("OpenCL LK: empty image");
        }
        if (gray.type() != CV_8UC1) {
            throw std::runtime_error("OpenCL LK currently expects CV_8UC1 grayscale image");
        }
        return gray.isContinuous() ? gray : gray.clone();
    }

    std::vector<ClImage> buildPyramid(const cv::Mat& gray, int max_level) const
    {
        std::vector<ClImage> pyramid(static_cast<size_t>(max_level + 1));
        const int count = gray.rows * gray.cols;

        ClBuffer input = makeBuffer(static_cast<size_t>(count), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gray.data);
        pyramid[0].rows = gray.rows;
        pyramid[0].cols = gray.cols;
        pyramid[0].data = makeBuffer(static_cast<size_t>(count) * sizeof(float), CL_MEM_READ_WRITE);

        cl_mem input_mem = input.mem;
        cl_mem level0_mem = pyramid[0].data.mem;
        run1D("u8_to_float", static_cast<size_t>(count), {
            {sizeof(cl_mem), &input_mem},
            {sizeof(cl_mem), &level0_mem},
            {sizeof(int), &count},
        });

        for (int level = 1; level <= max_level; ++level) {
            const ClImage& prev = pyramid[static_cast<size_t>(level - 1)];
            if (prev.rows < 2 || prev.cols < 2) {
                throw std::runtime_error("OpenCL LK: image too small for pyramid");
            }
            ClImage next;
            next.rows = std::max(1, prev.rows / 2);
            next.cols = std::max(1, prev.cols / 2);
            next.data = makeBuffer(
                static_cast<size_t>(next.rows) * static_cast<size_t>(next.cols) * sizeof(float),
                CL_MEM_READ_WRITE
            );

            cl_mem src_mem = prev.data.mem;
            cl_mem dst_mem = next.data.mem;
            run2D("blur_downsample", static_cast<size_t>(next.cols), static_cast<size_t>(next.rows), {
                {sizeof(cl_mem), &src_mem},
                {sizeof(cl_mem), &dst_mem},
                {sizeof(int), &prev.rows},
                {sizeof(int), &prev.cols},
                {sizeof(int), &next.rows},
                {sizeof(int), &next.cols},
            });

            pyramid[static_cast<size_t>(level)] = std::move(next);
        }

        return pyramid;
    }

    std::vector<ClImage> computeGradients(const std::vector<ClImage>& pyramid) const
    {
        std::vector<ClImage> gradients(pyramid.size() * 2);
        for (size_t level = 0; level < pyramid.size(); ++level) {
            const ClImage& img = pyramid[level];
            ClImage gx;
            ClImage gy;
            gx.rows = gy.rows = img.rows;
            gx.cols = gy.cols = img.cols;
            const size_t bytes = static_cast<size_t>(img.rows) * static_cast<size_t>(img.cols) * sizeof(float);
            gx.data = makeBuffer(bytes, CL_MEM_READ_WRITE);
            gy.data = makeBuffer(bytes, CL_MEM_READ_WRITE);

            cl_mem img_mem = img.data.mem;
            cl_mem gx_mem = gx.data.mem;
            cl_mem gy_mem = gy.data.mem;
            run2D("sobel_gradients", static_cast<size_t>(img.cols), static_cast<size_t>(img.rows), {
                {sizeof(cl_mem), &img_mem},
                {sizeof(cl_mem), &gx_mem},
                {sizeof(cl_mem), &gy_mem},
                {sizeof(int), &img.rows},
                {sizeof(int), &img.cols},
            });

            gradients[level * 2] = std::move(gx);
            gradients[level * 2 + 1] = std::move(gy);
        }
        return gradients;
    }

    void runOpenCl(
        const cv::Mat& prev_gray_raw,
        const cv::Mat& curr_gray_raw,
        const std::vector<cv::Point2f>& pts0,
        const std::vector<cv::Point2f>* initial_guess,
        std::vector<cv::Point2f>& pts1,
        std::vector<uchar>& status,
        std::vector<float>& err,
        int win_size,
        int max_level,
        int max_iters,
        float eps
    ) const {
        if (win_size <= 1 || (win_size % 2) == 0) {
            throw std::runtime_error("OpenCL LK: win_size must be odd and > 1");
        }
        if (max_level < 0 || max_iters <= 0 || eps <= 0.0f) {
            throw std::runtime_error("OpenCL LK: invalid LK parameters");
        }
        if (initial_guess != nullptr && initial_guess->size() != pts0.size()) {
            throw std::runtime_error("OpenCL LK: pts0 and initial_guess size mismatch");
        }

        const size_t count = pts0.size();
        pts1.resize(count);
        status.assign(count, 0);
        err.assign(count, -1.0f);
        if (count == 0) {
            return;
        }

        const cv::Mat prev_gray = continuousGray(prev_gray_raw);
        const cv::Mat curr_gray = continuousGray(curr_gray_raw);
        if (prev_gray.size() != curr_gray.size()) {
            throw std::runtime_error("OpenCL LK: image sizes must match");
        }

        std::vector<ClImage> prev_pyr = buildPyramid(prev_gray, max_level);
        std::vector<ClImage> curr_pyr = buildPyramid(curr_gray, max_level);
        std::vector<ClImage> curr_grad = computeGradients(curr_pyr);

        ClBuffer pts0_buf = makeBuffer(count * sizeof(cv::Point2f), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pts0.data());
        ClBuffer pts1_buf = makeBuffer(count * sizeof(cv::Point2f), CL_MEM_WRITE_ONLY);
        ClBuffer flow_buf = makeBuffer(count * sizeof(cv::Point2f), CL_MEM_READ_WRITE);
        ClBuffer status_buf = makeBuffer(count * sizeof(uchar), CL_MEM_READ_WRITE);
        ClBuffer err_buf = makeBuffer(count * sizeof(float), CL_MEM_READ_WRITE);

        const int count_i = static_cast<int>(count);
        cl_mem flow_mem = flow_buf.mem;
        cl_mem status_mem = status_buf.mem;
        cl_mem err_mem = err_buf.mem;

        if (initial_guess == nullptr) {
            run1D("init_flow_zero", count, {
                {sizeof(cl_mem), &flow_mem},
                {sizeof(cl_mem), &status_mem},
                {sizeof(cl_mem), &err_mem},
                {sizeof(int), &count_i},
            });
        } else {
            ClBuffer guess_buf = makeBuffer(
                count * sizeof(cv::Point2f),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                initial_guess->data()
            );
            cl_mem pts0_mem = pts0_buf.mem;
            cl_mem guess_mem = guess_buf.mem;
            const float inv_top_scale = 1.0f / static_cast<float>(1 << max_level);
            run1D("init_flow_guess", count, {
                {sizeof(cl_mem), &pts0_mem},
                {sizeof(cl_mem), &guess_mem},
                {sizeof(cl_mem), &flow_mem},
                {sizeof(cl_mem), &status_mem},
                {sizeof(cl_mem), &err_mem},
                {sizeof(float), &inv_top_scale},
                {sizeof(int), &count_i},
            });
        }

        for (int level = max_level; level >= 0; --level) {
            const ClImage& prev = prev_pyr[static_cast<size_t>(level)];
            const ClImage& curr = curr_pyr[static_cast<size_t>(level)];
            const ClImage& gx = curr_grad[static_cast<size_t>(level) * 2];
            const ClImage& gy = curr_grad[static_cast<size_t>(level) * 2 + 1];
            const float level_scale = 1.0f / static_cast<float>(1 << level);

            cl_mem prev_mem = prev.data.mem;
            cl_mem curr_mem = curr.data.mem;
            cl_mem gx_mem = gx.data.mem;
            cl_mem gy_mem = gy.data.mem;
            cl_mem pts0_mem = pts0_buf.mem;

            run1D("track_level", count, {
                {sizeof(cl_mem), &prev_mem},
                {sizeof(cl_mem), &curr_mem},
                {sizeof(cl_mem), &gx_mem},
                {sizeof(cl_mem), &gy_mem},
                {sizeof(cl_mem), &pts0_mem},
                {sizeof(cl_mem), &flow_mem},
                {sizeof(cl_mem), &status_mem},
                {sizeof(cl_mem), &err_mem},
                {sizeof(int), &curr.rows},
                {sizeof(int), &curr.cols},
                {sizeof(float), &level_scale},
                {sizeof(int), &win_size},
                {sizeof(int), &max_iters},
                {sizeof(float), &eps},
                {sizeof(int), &count_i},
            });

            if (level > 0) {
                run1D("scale_flow", count, {
                    {sizeof(cl_mem), &flow_mem},
                    {sizeof(cl_mem), &status_mem},
                    {sizeof(int), &count_i},
                });
            }
        }

        cl_mem pts0_mem = pts0_buf.mem;
        cl_mem pts1_mem = pts1_buf.mem;
        run1D("finish_tracks", count, {
            {sizeof(cl_mem), &pts0_mem},
            {sizeof(cl_mem), &flow_mem},
            {sizeof(cl_mem), &pts1_mem},
            {sizeof(cl_mem), &status_mem},
            {sizeof(int), &count_i},
        });

        checkCl(clEnqueueReadBuffer(queue_, pts1_buf.mem, CL_TRUE, 0, count * sizeof(cv::Point2f), pts1.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(pts1)");
        checkCl(clEnqueueReadBuffer(queue_, status_buf.mem, CL_TRUE, 0, count * sizeof(uchar), status.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(status)");
        checkCl(clEnqueueReadBuffer(queue_, err_buf.mem, CL_TRUE, 0, count * sizeof(float), err.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(err)");
    }

    void handleRuntimeFailure(const std::exception& ex) const
    {
        if (strict_) {
            throw;
        }
        std::cerr << "OpenCL LK failed, falling back to CPU: " << ex.what() << "\n";
    }

    bool strict_ = false;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    std::string name_;
    CpuVisionComputeBackend cpu_fallback_{"OpenCL runtime fallback"};
};

#endif

} // namespace

std::shared_ptr<VisionComputeBackend> VisionComputeBackend::createAuto()
{
    const std::string mode = gpuModeFromEnv();
    if (mode == "off") {
        auto backend = std::make_shared<CpuVisionComputeBackend>("forced by VIO_GPU=off");
        std::cout << "Vision backend: " << backend->name() << "\n";
        return backend;
    }

#if defined(VIO_HAVE_OPENCL)
    try {
        auto backend = std::make_shared<OpenClVisionComputeBackend>(mode == "on");
        std::cout << "Vision backend: " << backend->name() << "\n";
        return backend;
    } catch (const std::exception& ex) {
        if (mode == "on") {
            throw;
        }
        auto backend = std::make_shared<CpuVisionComputeBackend>(ex.what());
        std::cout << "Vision backend: " << backend->name() << "\n";
        return backend;
    }
#else
    if (mode == "on") {
        throw std::runtime_error("VIO_GPU=on requested, but this build has no OpenCL support");
    }
    auto backend = std::make_shared<CpuVisionComputeBackend>("OpenCL not available in this build");
    std::cout << "Vision backend: " << backend->name() << "\n";
    return backend;
#endif
}

