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

void SurfDetector::detectAndCompute(
    cv::InputArray image, cv::InputArray mask,
    std::vector<cv::KeyPoint>& keypoints,
    cv::OutputArray descriptors,
    bool useProvidedKeypoints)
{
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
    ParallelSurf::Image psImg(gray);

    cvg::Surf surf(m_extended);

    std::vector<ParallelSurf::KeyPoint> psKps;
    surf.detect(psImg, psKps);

    keypoints.resize(psKps.size());
    for (size_t i = 0; i < psKps.size(); i++)
    {
        keypoints[i].pt.x = (float)psKps[i]._x;
        keypoints[i].pt.y = (float)psKps[i]._y;
        keypoints[i].size = (float)(psKps[i]._scale * 2.5f);
        keypoints[i].response = (float)psKps[i]._score;
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

    cv::Mat gray = toGray(image);
    ParallelSurf::Image psImg(gray);

    cvg::Surf surf(m_extended);

    // Convert cv::KeyPoint to ParallelSurf::KeyPoint
    std::vector<ParallelSurf::KeyPoint> psKps(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        psKps[i]._x = keypoints[i].pt.x;
        psKps[i]._y = keypoints[i].pt.y;
        psKps[i]._scale = keypoints[i].size / 2.5;
        psKps[i]._score = keypoints[i].response;
    }

    // Assign orientations and compute descriptors
    surf.assignOrientations(psImg, psKps.begin(), psKps.end());
    surf.compute(psImg, psKps);

    // Convert descriptors to cv::Mat
    int vecLen = psKps[0]._vec.size();
    cv::Mat descMat((int)psKps.size(), vecLen, CV_32FC1);
    for (size_t i = 0; i < psKps.size(); i++)
    {
        float* row = descMat.ptr<float>((int)i);
        for (int j = 0; j < vecLen; j++)
            row[j] = (float)psKps[i]._vec[j];
    }
    descMat.copyTo(descriptors);
}
