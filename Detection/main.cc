//
// Created by jerett on 2018-12-25.
//


/**
 * Test OpenCV DNN module, yolov3 detection.
 */

#include <fstream>
#include <vector>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "yolov3_detector.h"

using namespace std;
using namespace cv;
using namespace cd;

void DrawDetection(const std::string &name, double conf, const Rect &box, Mat &frame) {
    rectangle(frame, box, Scalar(0, 255, 0));
    std::string label = format("%.2f", conf);
    label = name + ": " + label;

    const Point top_left(box.x, box.y);
    int baseLine = 0;
    Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    Rect label_rect(top_left, Size(label_size.width, baseLine + label_size.height));
    rectangle(frame, label_rect, Scalar::all(255), FILLED);
    putText(frame, label, top_left + Point(0, label_size.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
}

void DrawDetections(const std::vector<ObjectDetection> &detections, cv::Mat &img) {
    for (auto &detection : detections) {
        CV_LOG_INFO(NULL, "detect obj:" << detection.name
                                        << ", box:" << detection.box
                                        << ", conf:" << detection.confidence);
        DrawDetection(detection.name, detection.confidence, detection.box, img);
    }
}


int main(int argc, char *argv[]) {
    const String keys =
        "{help h usage ? |      | print this message   }"
        "{classes        |<none>| path to a text file with names of classes to label detected objects. }"
        "{cfg            |<none>| yolo network cfg file. }"
        "{model m        |<none>| yolo network binary weights file. }"
        "{img  i         |<none>| img path }";
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::string img_path = parser.get<String>("img");
    std::string classes_txt = parser.get<String>("classes");
    std::string cfg_file = parser.get<String>("cfg");
    std::string model_file = parser.get<String>("model");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
    cd::YOLOV3Detector detector(classes_txt, cfg_file, model_file);

    if (!detector.IsOpen()) {
        CV_LOG_ERROR(NULL, "open detector failed.");
        return -1;
    }

    auto src = imread(img_path);
    if (src.empty()) {
        CV_LOG_ERROR(NULL, "read img error:" << img_path);
        return -1;
    }

    // Mat nms_img = dst.clone();
    auto detections = detector.Detect(src, true);
    DrawDetections(detections, src);
    cv::imshow("fuck", src);
    cv::waitKey(0);

    return 0;
}

