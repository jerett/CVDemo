//
// Created by jerett on 2019-01-03.
//

#pragma once

#include <limits>

template<typename T> inline
T calcPsr(const cv::Mat &response, const cv::Point2i &maxResponseIdx, const int deletionRange, T& peakValue)
{
    peakValue = response.at<T>(maxResponseIdx);
    double psrClamped = 0;

    cv::Mat sideLobe = response.clone();
    sideLobe.setTo(0, sideLobe < 0);

    cv::rectangle(sideLobe,
                  cv::Point2i(maxResponseIdx.x - deletionRange, maxResponseIdx.y - deletionRange),
                  cv::Point2i(maxResponseIdx.x + deletionRange, maxResponseIdx.y + deletionRange),
                  cv::Scalar(0), cv::FILLED);

    cv::Scalar mean_;
    cv::Scalar std_;
    cv::meanStdDev(sideLobe, mean_, std_);
    const static T eps = std::numeric_limits<T>::epsilon();
    psrClamped = (peakValue - mean_[0]) / (std_[0] + eps);
    return static_cast<T>(psrClamped);
}

