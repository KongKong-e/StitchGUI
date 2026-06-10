#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dlldefine.h"


using namespace cv::detail;



class  SUPERSTITCH_EXPORTS LightGlue :public FeaturesMatcher
{
protected:
	cv::Stitcher::Mode m_mode;

	std::wstring m_modelPath;

	std::vector<cv::detail::ImageFeatures> features_;

	std::vector<cv::detail::MatchesInfo> pairwise_matches_;

	float m_matchThresh = 0.0;

	bool m_useGpu = false;
	bool m_gpuActive = false;
	bool m_isEnglish = false;

	CV_WRAP_AS(apply) void operator ()(
		const ImageFeatures& features1,
		const ImageFeatures& features2,
		CV_OUT MatchesInfo& matches_info)
	{
		match(features1, features2, matches_info);
	}

	void AddFeature(cv::detail::ImageFeatures features);

	void AddMatcheinfo(const MatchesInfo& matches_info);

public:
	LightGlue(
		std::wstring modelPath,
		cv::Stitcher::Mode mode,
		float matchThresh,
		bool useGpu = false,
		bool isEnglish = false);

	void match(
		const ImageFeatures& features1,
		const ImageFeatures& features2,
		MatchesInfo& matches_info);

	std::vector<cv::detail::ImageFeatures> features()
	{
		return features_;
	};


	std::vector<cv::detail::MatchesInfo> matchinfo()
	{
		return pairwise_matches_;
	};

	void clearCache()
	{
		features_.clear();
		features_.shrink_to_fit();
		pairwise_matches_.clear();
		pairwise_matches_.shrink_to_fit();
	};

	// 实际是否使用 GPU（session 创建后才准确）
	bool isUsingGpu() const { return m_gpuActive; }

};
