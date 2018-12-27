//
// Created by jerett on 2018-12-26.
//

/**
 *  implement yolov3 detector with OpenCV dnn module.
 */

#pragma once

#include <string>
#include <opencv2/dnn.hpp>

namespace cd {

struct ObjectDetection {
    std::string name;
    float confidence;
    cv::Rect box;
};

class YOLOV3Detector {

public:
    YOLOV3Detector(const std::string &class_txt,
                   const std::string &cfg_file,
                   const std::string &weights_file,
                   cv::Size intput_size = cv::Size(608, 608),
                   float conf_threshold = 0.5,
                   float nms_threshold = 0.4);

    bool IsOpen() const {
        return is_open_;
    }

    std::vector<ObjectDetection> Detect(const cv::Mat &img, bool nms=true);

private:
    void LoadClasses(const std::string &class_txt);

private:
    cv::Size input_size_;
    const float conf_threshold_;
    const float nms_threshold_;
    bool is_open_ = false;
    std::vector<std::string> classes_;  // class names
    cv::dnn::Net net_;

};

}


