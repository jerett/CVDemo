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

void ShowResult(cv::Mat &origin_img, cv::Mat &raw_detection_img, cv::Mat &nms_detection_img) {
    cv::putText(origin_img, "origin", Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    cv::putText(raw_detection_img, "raw detection", Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    cv::putText(nms_detection_img, "nms detection", Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    cv::Mat mat_arr[] = {
        origin_img, raw_detection_img, nms_detection_img,
    };
    Mat display_img;
    hconcat(mat_arr, 3, display_img);
    // imshow("origin", img);
    // imshow("nms detection", nms_img);
    // imshow("no nms detection", no_nms_img);
    imshow("detection", display_img);
    waitKey(0);
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

    // cd::YOLOV3Detector detector("coco.names", "yolov3.cfg", "yolov3.weights");
    cd::YOLOV3Detector detector(classes_txt, cfg_file, model_file);
    if (!detector.IsOpen()) {
        CV_LOG_ERROR(NULL, "open detector failed.");
        return -1;
    }

    auto img = imread(img_path);
    if (img.empty()) {
        CV_LOG_ERROR(NULL, "read img error:" << img_path);
        return -1;
    }

    Mat nms_img = img.clone();
    const auto nms_detections = detector.Detect(img, true);
    DrawDetections(nms_detections, nms_img);

    Mat no_nms_img = img.clone();
    const auto no_nms_detections = detector.Detect(img, false);
    DrawDetections(no_nms_detections, no_nms_img);

    ShowResult(img, no_nms_img, nms_img);
    cv::imwrite("result.jpg", nms_img);
    return 0;
}

