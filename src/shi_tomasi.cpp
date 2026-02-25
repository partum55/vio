#include "shi_tomasi.hpp"

static cv::Mat toGrayFloat01(const cv::Mat &imgBgrOrGray)
{
    cv::Mat gray;
    if (imgBgrOrGray.channels() == 3)
    {
        cv::cvtColor(imgBgrOrGray, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = imgBgrOrGray;
    }
    cv::Mat gray32;
    gray.convertTo(gray32, CV_32F, 1.0 / 255.0);
    return gray32;
}

// lambda_min for each pixel
static cv::Mat shiTomasiScoreImage(const cv::Mat &gray32, const ShiTomasiParams &p)
{
    cv::Mat blur;
    cv::GaussianBlur(gray32, blur, cv::Size(0, 0), p.gaussianSigma, p.gaussianSigma);

    cv::Mat Ix, Iy;
    cv::Sobel(blur, Ix, CV_32F, 1, 0, p.sobelKSize);
    cv::Sobel(blur, Iy, CV_32F, 0, 1, p.sobelKSize);

    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // sum over patch
    // blockSize is the patch size
    double sigmaTensor = std::max(0.5, p.blockSize / 3.0);
    cv::GaussianBlur(Ixx, Ixx, cv::Size(0, 0), sigmaTensor, sigmaTensor);
    cv::GaussianBlur(Iyy, Iyy, cv::Size(0, 0), sigmaTensor, sigmaTensor);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(0, 0), sigmaTensor, sigmaTensor);

    // lambda_min = tr/2 - sqrt((tr/2)^2 - det)
    cv::Mat trace = Ixx + Iyy;
    cv::Mat det = Ixx.mul(Iyy) - Ixy.mul(Ixy);

    cv::Mat halfTrace = 0.5f * trace;
    cv::Mat inside = halfTrace.mul(halfTrace) - det;

    // numerical stability
    cv::max(inside, 0, inside);

    cv::Mat sqrtInside;
    cv::sqrt(inside, sqrtInside);

    cv::Mat score = halfTrace - sqrtInside;
    cv::max(score, 0, score);

    return score;
}

// NMS - non-maximum suppression: keep only local max
static cv::Mat nmsLocalMax(const cv::Mat &score, int r)
{
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * r + 1, 2 * r + 1));
    cv::dilate(score, dilated, kernel);

    cv::Mat isMax = (score == dilated);
    // zero-out non-max
    cv::Mat out = cv::Mat::zeros(score.size(), score.type());
    score.copyTo(out, isMax);
    return out;
}

std::vector<cv::Point2f> extractShiTomasiKeypoints(const cv::Mat &img, const ShiTomasiParams &p)
{
    cv::Mat gray32 = toGrayFloat01(img);
    cv::Mat score = shiTomasiScoreImage(gray32, p);
    cv::Mat scoreNms = nmsLocalMax(score, p.nmsRadius);

    double minVal, maxVal;
    cv::minMaxLoc(scoreNms, &minVal, &maxVal);
    const auto thresh = static_cast<float>(p.qualityLevel * maxVal);

    // collect candidates
    std::vector<Candidate> cand;
    cand.reserve(scoreNms.rows * scoreNms.cols / 20);

    for (int y = 0; y < scoreNms.rows; ++y)
    {
        const float *row = scoreNms.ptr<float>(y);
        for (int x = 0; x < scoreNms.cols; ++x)
        {
            float s = row[x];
            if (s >= thresh)
                cand.push_back({s, x, y});
        }
    }

    // sort by score desc
    std::sort(cand.begin(), cand.end(), [](const Candidate &a, const Candidate &b)
              { return a.score > b.score; });

    // greedy minDistance selection
    std::vector<cv::Point2f> pts;
    pts.reserve(std::min<int>(p.maxCorners, (int)cand.size()));

    const auto minDist2 = static_cast<float>(p.minDistance * p.minDistance);

    for (const auto &c : cand)
    {
        if ((int)pts.size() >= p.maxCorners)
            break;

        cv::Point2f pt(
            static_cast<float>(c.x),
            static_cast<float>(c.y)
        );

        bool ok = true;
        for (const auto &chosen : pts)
        {
            float dx = pt.x - chosen.x;
            float dy = pt.y - chosen.y;
            if (dx * dx + dy * dy < minDist2)
            {
                ok = false;
                break;
            }
        }
        if (ok)
            pts.push_back(pt);
    }

    return pts;
}

cv::Mat toGrayU8(const cv::Mat &imgBgrOrGray)
{
    cv::Mat gray;
    if (imgBgrOrGray.channels() == 3)
    {
        cv::cvtColor(imgBgrOrGray, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = imgBgrOrGray;
    }
    if (gray.type() != CV_8U)
    {
        cv::Mat tmp;
        gray.convertTo(tmp, CV_8U);
        return tmp;
    }
    return gray;
}

cv::Mat drawKeypointsOnImage(const cv::Mat &imgBgrOrGray,
                             const std::vector<cv::Point2f> &pts,
                             int radius,
                             int thickness)
{
    cv::Mat vis;
    if (imgBgrOrGray.channels() == 1)
    {
        cv::cvtColor(imgBgrOrGray, vis, cv::COLOR_GRAY2BGR);
    }
    else
    {
        vis = imgBgrOrGray.clone();
    }

    for (const auto &p : pts)
    {
        cv::circle(vis, p, radius, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
    }
    return vis;
}
