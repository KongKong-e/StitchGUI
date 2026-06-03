#include "classical_matcher.h"
#include <QDebug>

ClassicalMatcher::ClassicalMatcher(cv::Stitcher::Mode mode, float matchThresh)
    : FeaturesMatcher(false)
    , m_mode(mode)
    , m_matchThresh(matchThresh)
{
}

void ClassicalMatcher::match(
    const cv::detail::ImageFeatures& features1,
    const cv::detail::ImageFeatures& features2,
    cv::detail::MatchesInfo& matches_info)
{
    matches_info.src_img_idx = features1.img_idx;
    matches_info.dst_img_idx = features2.img_idx;
    matches_info.H = cv::Mat();
    matches_info.confidence = 0;
    matches_info.num_inliers = 0;

    if (features1.keypoints.empty() || features2.keypoints.empty())
        return;

    // KNN 匹配 (k=2)，用于 Lowe's ratio test
    std::vector<std::vector<cv::DMatch>> knnMatches;

    if (features1.descriptors.depth() == CV_8U)
    {
        // ORB 等 binary 描述子：BFMatcher + 汉明距离
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.knnMatch(features1.descriptors, features2.descriptors, knnMatches, 2);
    }
    else
    {
        // SIFT/SuperPoint/SURF 等 float 描述子：BFMatcher + L2 距离
        // 注意：FLANN KD-Tree 在某些 OpenCV 构建中处理 SURF 描述子会导致堆损坏
        cv::BFMatcher matcher(cv::NORM_L2);
        matcher.knnMatch(features1.descriptors, features2.descriptors, knnMatches, 2);
    }

    // Lowe's ratio test 筛选
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        if (knnMatches[i].size() == 2 &&
            knnMatches[i][0].distance < (1.f - m_matchThresh) * knnMatches[i][1].distance)
        {
            matches_info.matches.push_back(knnMatches[i][0]);
        }
    }

    if (matches_info.matches.size() < 6) {
        qDebug("BFMatcher: [%d vs %d] Ratio test 后匹配数不足: %d < 6, 跳过",
            features1.img_idx, features2.img_idx, (int)matches_info.matches.size());
        return;
    }

    qDebug("BFMatcher: [%d vs %d] Ratio test 后匹配数: %d",
        features1.img_idx, features2.img_idx, (int)matches_info.matches.size());

    // 估计几何变换（坐标需以图像中心为原点，同 LightGlue 做法）
    cv::Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    cv::Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);

    if (m_mode == cv::Stitcher::SCANS)
    {
        for (size_t i = 0; i < matches_info.matches.size(); ++i)
        {
            src_points.at<cv::Point2f>(0, static_cast<int>(i)) =
                features1.keypoints[matches_info.matches[i].queryIdx].pt;
            dst_points.at<cv::Point2f>(0, static_cast<int>(i)) =
                features2.keypoints[matches_info.matches[i].trainIdx].pt;
        }

        matches_info.H = cv::estimateAffine2D(src_points, dst_points, matches_info.inliers_mask);

        if (matches_info.H.empty())
        {
            matches_info.confidence = 0;
            matches_info.num_inliers = 0;
            return;
        }

        matches_info.num_inliers = 0;
        for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
        {
            if (matches_info.inliers_mask[i])
                matches_info.num_inliers++;
        }

        matches_info.confidence =
            matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

        matches_info.H.push_back(cv::Mat::zeros(1, 3, CV_64F));
        matches_info.H.at<double>(2, 2) = 1;
    }
    else if (m_mode == cv::Stitcher::PANORAMA)
    {
        // 使用原始图像坐标（不中心化），cv::Stitcher 期望 Homography 在原始坐标系下
        for (size_t i = 0; i < matches_info.matches.size(); ++i)
        {
            const cv::DMatch& m = matches_info.matches[i];
            src_points.at<cv::Point2f>(0, static_cast<int>(i)) = features1.keypoints[m.queryIdx].pt;
            dst_points.at<cv::Point2f>(0, static_cast<int>(i)) = features2.keypoints[m.trainIdx].pt;
        }

        matches_info.H = cv::findHomography(src_points, dst_points, matches_info.inliers_mask, cv::RANSAC);
        if (matches_info.H.empty() || std::abs(cv::determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
            return;

        matches_info.num_inliers = 0;
        for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
        {
            if (matches_info.inliers_mask[i])
                matches_info.num_inliers++;
        }

        matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

        qDebug("BFMatcher: [%d vs %d] 匹配数: %d, 内点: %d, 置信度: %.3f",
            features1.img_idx, features2.img_idx,
            (int)matches_info.matches.size(), matches_info.num_inliers, matches_info.confidence);

        if (matches_info.num_inliers < 6)
            return;

        // 二次 RANSAC 精炼（仅用内点重估，原始坐标）
        src_points.create(1, matches_info.num_inliers, CV_32FC2);
        dst_points.create(1, matches_info.num_inliers, CV_32FC2);
        int inlier_idx = 0;
        for (size_t i = 0; i < matches_info.matches.size(); ++i)
        {
            if (!matches_info.inliers_mask[i])
                continue;

            const cv::DMatch& m = matches_info.matches[i];
            src_points.at<cv::Point2f>(0, inlier_idx) = features1.keypoints[m.queryIdx].pt;
            dst_points.at<cv::Point2f>(0, inlier_idx) = features2.keypoints[m.trainIdx].pt;
            inlier_idx++;
        }

        matches_info.H = cv::findHomography(src_points, dst_points, cv::RANSAC);
    }

    // 缓存数据用于可视化
    this->AddFeature(features1);
    this->AddFeature(features2);
    this->AddMatcheinfo(matches_info);
}

void ClassicalMatcher::AddFeature(cv::detail::ImageFeatures features)
{
    bool found = false;
    for (size_t i = 0; i < this->features_.size(); i++)
    {
        if (features.img_idx == this->features_[i].img_idx)
        {
            found = true;
            break;
        }
    }

    if (!found)
        this->features_.push_back(features);
}

void ClassicalMatcher::AddMatcheinfo(const cv::detail::MatchesInfo& matches)
{
    bool found = false;

    for (size_t i = 0; i < this->pairwise_matches_.size(); i++)
    {
        if (matches.src_img_idx == this->pairwise_matches_[i].src_img_idx &&
            matches.dst_img_idx == this->pairwise_matches_[i].dst_img_idx)
        {
            found = true;
            break;
        }

        if (matches.src_img_idx == this->pairwise_matches_[i].dst_img_idx &&
            matches.dst_img_idx == this->pairwise_matches_[i].src_img_idx)
        {
            found = true;
            break;
        }
    }

    if (!found)
        this->pairwise_matches_.push_back(cv::detail::MatchesInfo(matches));
}
