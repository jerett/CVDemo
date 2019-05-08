//
// Created by jerett on 2019-03-18.
//

#pragma once

#include <cmath>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace cd {

static float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

static void softmax(std::vector<float> &classes) {
    float sum = 0;
    std::transform(classes.begin(), classes.end(), classes.begin(),
                   [&sum](float score) -> float {
                       float exp_score = exp(score);
                       sum += exp_score;
                       return exp_score;
                   });
    std::transform(classes.begin(), classes.end(), classes.begin(),
                   [sum](float score) -> float { return score / sum; });
}

cv::Mat DetectPreprocess(const cv::Mat &in_img,
                         const cv::Size &net_size,
                         float *scale, float *pad_w, float *pad_h);

cv::Rect2d RemapBoxOnSrc(const cv::Rect2d &box, const cv::Size &net_size, const cv::Size &img_size,
                         float scale, float pad_w, float pad_h);

}