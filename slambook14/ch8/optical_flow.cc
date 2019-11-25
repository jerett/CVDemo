//
// Created by jerett on 2019/11/22.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char *argv[]) {
    const Mat img1 = imread("LK1.png", IMREAD_GRAYSCALE);
    const Mat img2 = imread("LK2.png", IMREAD_GRAYSCALE);

    auto detector = GFTTDetector::create(500, 0.01, 20);
    std::vector<KeyPoint> img1_kpts;
    detector->detect(img1, img1_kpts);
    std::vector<uchar> status;
    std::vector<float> error;

    std::vector<Point2f> img1_pts;
    for (const auto &kp : img1_kpts) {
        img1_pts.push_back(kp.pt);
    }
    std::vector<Point2f> img2_pts;
    calcOpticalFlowPyrLK(img1, img2, img1_pts, img2_pts, status, error);

    Mat img2_draw;
    cvtColor(img2, img2_draw, COLOR_GRAY2BGR);
    for (int i = 0; i < img2_pts.size(); ++i) {
        if (status[i]) {
            circle(img2_draw, img2_pts[i], 2, Scalar(0, 250, 0), 2);
            line(img2_draw, img1_pts[i], img2_pts[i], Scalar(0, 250, 0), 1);
        }
    }
    imshow("img2 draw", img2_draw);
    waitKey(0);

    return 0;
}