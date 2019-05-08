//
// Created by jerett on 2019-02-28.
//

#pragma once

#include <opencv2/opencv.hpp>

namespace cd {

struct ObjectDetection {
    std::string name;
    float confidence;
    cv::Rect box;
};

}
