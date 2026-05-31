#include "surf_wrapper.h"
#include "Surf.h"

static cv::Mat toGray(const cv::InputArray& image)
{
    cv::Mat img = image.getMat();
    cv::Mat gray;
    if (img.channels() >= 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();
    return gray;
}

static void parallelsurfToCv(
    const std::vector<ParallelSurf::KeyPoint>& psKps,
    std::vector<cv::KeyPoint>& cvKps,
    cv::Mat& descMat)
{
    cvKps.resize(psKps.size());

    int vecLen = psKps.empty() ? 0 : (int)psKps[0]._vec.size();

    if (vecLen > 0)
        descMat.create((int)psKps.size(), vecLen, CV_32FC1);

    for (size_t i = 0; i < psKps.size(); i++)
    {
        const ParallelSurf::KeyPoint& kp = psKps[i];
        cvKps[i].pt.x = (float)kp._x;
        cvKps[i].pt.y = (float)kp._y;
        cvKps[i].size = (float)(kp._scale * 2.5f);
        cvKps[i].response = (float)kp._score;
        cvKps[i].angle = (float)(kp._ori * 180.0 / CV_PI);

        if (vecLen > 0)
        {
            float* row = descMat.ptr<float>((int)i);
            for (int j = 0; j < vecLen; j++)
                row[j] = (float)kp._vec[j];
        }
    }
}

SurfDetector::SurfDetector(bool extended)
    : m_extended(extended)
{
}

int64_t SurfDetector::computeImageHash(const cv::Mat& img)
{
    // 对图像缩放到 8x8 灰度，转为二进制 hash（感知哈希）
    // 用于在 detect()/compute() 之间快速匹配同一张图像
    cv::Mat small;
    if (img.channels() >= 3)
        cv::cvtColor(img, small, cv::COLOR_BGR2GRAY);
    else
        small = img;

    cv::Mat resized;
    cv::resize(small, resized, cv::Size(8, 8));
    resized.convertTo(resized, CV_64F);

    double mean = cv::mean(resized)[0];
    int64_t hash = 0;
    for (int i = 0; i < 8; i++)
    {
        const double* row = resized.ptr<double>(i);
        for (int j = 0; j < 8; j++)
        {
            if (row[j] >= mean)
                hash |= (1LL << (i * 8 + j));
        }
    }
    return hash;
}

void SurfDetector::detectAndCompute(
    cv::InputArray image, cv::InputArray mask,
    std::vector<cv::KeyPoint>& keypoints,
    cv::OutputArray descriptors,
    bool useProvidedKeypoints)
{
    m_cache.clear();

    cv::Mat gray = toGray(image);
    ParallelSurf::Image psImg(gray);

    cvg::Surf surf(m_extended);
    std::vector<ParallelSurf::KeyPoint> psKps;

    surf.detect(psImg, psKps);

    if (psKps.empty())
    {
        keypoints.clear();
        descriptors.release();
        return;
    }

    surf.compute(psImg, psKps);

    cv::Mat descMat;
    parallelsurfToCv(psKps, keypoints, descMat);
    descMat.copyTo(descriptors);
}

void SurfDetector::detect(
    cv::InputArray image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::InputArray mask)
{
    cv::Mat gray = toGray(image);

    // 缓存灰度图和哈希，供后续 compute() 使用
    CachedData cache;
    cache.gray = gray.clone();
    cache.hash = computeImageHash(image.getMat());

    ParallelSurf::Image psImg(gray);
    cvg::Surf surf(m_extended);

    surf.detect(psImg, cache.psKps);

    m_cache.push_back(std::move(cache));

    const auto& psKps = m_cache.back().psKps;
    keypoints.resize(psKps.size());
    for (size_t i = 0; i < psKps.size(); i++)
    {
        keypoints[i].pt.x = (float)psKps[i]._x;
        keypoints[i].pt.y = (float)psKps[i]._y;
        keypoints[i].size = (float)(psKps[i]._scale * 2.5f);
        keypoints[i].response = (float)psKps[i]._score;
        keypoints[i].angle = (float)(psKps[i]._ori * 180.0 / CV_PI);
    }
}

void SurfDetector::compute(
    cv::InputArray image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::OutputArray descriptors)
{
    if (keypoints.empty())
    {
        descriptors.release();
        return;
    }

    // 通过图像哈希匹配 detect() 阶段缓存的数据
    // cv::Stitcher 先对所有图像调用 detect()，再调用 compute()
    // 必须匹配正确的缓存条目，否则关键点坐标与图像不对应
    int64_t hash = computeImageHash(image.getMat());
    int cacheIdx = -1;
    for (int i = 0; i < (int)m_cache.size(); i++)
    {
        if (m_cache[i].hash == hash &&
            m_cache[i].psKps.size() == keypoints.size())
        {
            cacheIdx = i;
            break;
        }
    }

    if (cacheIdx >= 0)
    {
        // 使用缓存的灰度图和关键点，确保坐标空间一致
        ParallelSurf::Image psImg(m_cache[cacheIdx].gray);
        cvg::Surf surf(m_extended);
        auto& psKps = m_cache[cacheIdx].psKps;

        try {
            surf.makeDescriptors(psImg, psKps.begin(), psKps.end());
        } catch (...) {
            descriptors.release();
            return;
        }

        int vecLen = psKps.empty() ? 0 : (int)psKps[0]._vec.size();
        if (vecLen == 0) {
            descriptors.release();
            return;
        }

        cv::Mat descMat((int)psKps.size(), vecLen, CV_32FC1);
        for (size_t i = 0; i < psKps.size(); i++)
        {
            float* row = descMat.ptr<float>((int)i);
            for (int j = 0; j < vecLen; j++)
                row[j] = (float)psKps[i]._vec[j];
        }
        descMat.copyTo(descriptors);

        // 已使用，清除该缓存条目
        m_cache.erase(m_cache.begin() + cacheIdx);
        return;
    }

    // 回退方案：缓存未命中，从 cv::KeyPoint 重建
    cv::Mat gray = toGray(image);
    ParallelSurf::Image psImg(gray);
    cvg::Surf surf(m_extended);

    std::vector<ParallelSurf::KeyPoint> psKps(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        psKps[i]._x = keypoints[i].pt.x;
        psKps[i]._y = keypoints[i].pt.y;
        psKps[i]._scale = keypoints[i].size / 2.5;
        psKps[i]._score = keypoints[i].response;
        psKps[i]._trace = 0;
        psKps[i]._ori = keypoints[i].angle * CV_PI / 180.0;
    }

    try {
        surf.makeDescriptors(psImg, psKps.begin(), psKps.end());
    } catch (...) {
        descriptors.release();
        return;
    }

    int vecLen = psKps.empty() ? 0 : (int)psKps[0]._vec.size();
    if (vecLen == 0) {
        descriptors.release();
        return;
    }

    cv::Mat descMat((int)psKps.size(), vecLen, CV_32FC1);
    for (size_t i = 0; i < psKps.size(); i++)
    {
        float* row = descMat.ptr<float>((int)i);
        for (int j = 0; j < vecLen; j++)
            row[j] = (float)psKps[i]._vec[j];
    }
    descMat.copyTo(descriptors);
}
