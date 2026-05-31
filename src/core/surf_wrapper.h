#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>
#include "dlldefine.h"
#include "parallelsurf/KeyPoint.h"

class SUPERSTITCH_EXPORTS SurfDetector : public cv::Feature2D
{
public:
    SurfDetector(bool extended = false);

    virtual void detectAndCompute(
        cv::InputArray image, cv::InputArray mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors,
        bool useProvidedKeypoints = false) override;

    virtual void detect(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray mask = cv::noArray()) override;

    virtual void compute(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors) override;

private:
    bool m_extended;

    // 缓存 detect() 阶段的数据，供 compute() 复用
    // cv::Stitcher 先对所有图像调用 detect()，再对所有图像调用 compute()
    // 因此需要缓存所有图像的数据，并在 compute() 时通过图像哈希匹配
    struct CachedData {
        cv::Mat gray;
        std::vector<ParallelSurf::KeyPoint> psKps;
        int64_t hash;
    };
    std::vector<CachedData> m_cache;

    // 计算图像的快速哈希，用于匹配 detect/compute 调用
    static int64_t computeImageHash(const cv::Mat& img);
};
