#pragma warning(disable:4996)

#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include <opencv2/opencv.hpp>


#include "dlldefine.h"

using namespace cv::detail;


class  SUPERSTITCH_EXPORTS SuperPoint :public cv::Feature2D
{
protected:

	std::wstring mModelPath;

	std::vector<float> ApplyTransform(
		const cv::Mat& image,
		float& mean, float& std);

public:
	SuperPoint(std::wstring modelPath);

	virtual void detectAndCompute(
		cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors,
		bool useProvidedKeypoints = false);

	virtual void detect(
		cv::InputArray image,
		std::vector<cv::KeyPoint>& keypoints,
		cv::InputArray mask = cv::noArray());

	virtual void compute(
		cv::InputArray image,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors);

};
