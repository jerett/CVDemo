//
// Created by jerett on 2018-12-26.
//


#include "yolov3_detector.h"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>

using namespace cv;

namespace cd {

YOLOV3Detector::YOLOV3Detector(const std::string &class_txt,
                               const std::string &cfg_file,
                               const std::string &weights_file,
                               cv::Size input_size,
                               float conf_threshold,
                               float nms_threshold)
    : input_size_(input_size), conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
    LoadClasses(class_txt);
    net_ = dnn::readNetFromDarknet(cfg_file, weights_file);
    net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
    is_open_ = true;
}

void YOLOV3Detector::LoadClasses(const std::string &class_txt) {
    std::ifstream ifs(class_txt.c_str());
    std::string line;
    while (getline(ifs, line)) {
        classes_.push_back(line);
    }
}

std::vector<ObjectDetection> YOLOV3Detector::Detect(const Mat &img, bool nms) {
    const double scale = 1.0 / 255;
    Scalar mean_value(0, 0, 0);
    cv::Mat blob = dnn::blobFromImage(img, scale, input_size_, mean_value, true, false, CV_32F);

    net_.setInput(blob);
    std::vector<Mat> outs;
    net_.forward(outs, net_.getUnconnectedOutLayersNames());

    static std::vector<int> outLayers = net_.getUnconnectedOutLayers();


    std::vector<int> ids;
    std::vector<Rect> boxes;
    std::vector<float> confidences;
    for (size_t i = 0; i < outs.size(); ++i) {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point max_confidence_id;
            double confidence;
            minMaxLoc(scores, nullptr, &confidence, nullptr, &max_confidence_id);
            if (confidence > conf_threshold_) {
                int centerX = (int) (data[0] * img.cols);
                int centerY = (int) (data[1] * img.rows);
                int width = (int) (data[2] * img.cols);
                int height = (int) (data[3] * img.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                ids.push_back(max_confidence_id.x);
                boxes.emplace_back(left, top, width, height);
                confidences.push_back(static_cast<float>(confidence));
            }
        }
    }

    std::vector<ObjectDetection> detections;
    std::vector<int> indices;
    if (nms) {
        dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
    } else {
        // if no nms, push all detections.
        for (int i = 0; i < boxes.size(); ++i) {
            indices.push_back(i);
        }
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];

        ObjectDetection object_detection;
        object_detection.box = box;
        object_detection.name = classes_[ids[idx]];
        object_detection.confidence = confidences[idx];
        detections.push_back(std::move(object_detection));
    }
    return detections;
}

}