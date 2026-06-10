#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "dlldefine.h"

class SUPERSTITCH_EXPORTS ClassicalMatcher : public cv::detail::FeaturesMatcher
{
protected:
    cv::Stitcher::Mode m_mode;
    float m_matchThresh;
    bool m_isEnglish;

    std::vector<cv::detail::ImageFeatures> features_;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_;

    CV_WRAP_AS(apply) void operator ()(
        const cv::detail::ImageFeatures& features1,
        const cv::detail::ImageFeatures& features2,
        CV_OUT cv::detail::MatchesInfo& matches_info)
    {
        match(features1, features2, matches_info);
    }

    void AddFeature(cv::detail::ImageFeatures features);
    void AddMatcheinfo(const cv::detail::MatchesInfo& matches_info);

public:
    ClassicalMatcher(cv::Stitcher::Mode mode, float matchThresh, bool isEnglish = false);

    void match(
        const cv::detail::ImageFeatures& features1,
        const cv::detail::ImageFeatures& features2,
        cv::detail::MatchesInfo& matches_info);

    std::vector<cv::detail::ImageFeatures> features()
    {
        return features_;
    }

    std::vector<cv::detail::MatchesInfo> matchinfo()
    {
        return pairwise_matches_;
    }

    void clearCache()
    {
        features_.clear();
        features_.shrink_to_fit();
        pairwise_matches_.clear();
        pairwise_matches_.shrink_to_fit();
    }
};
