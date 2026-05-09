#pragma once

#include <opencv2/opencv.hpp>
#include "dlldefine.h"

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
};
