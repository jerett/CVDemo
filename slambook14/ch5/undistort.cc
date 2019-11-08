//
// Created by jerett on 2019/11/5.
//

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

const std::string file("./distorted.png");

int main(int argc, char *argv[]) {
    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat distorted_img = cv::imread(file, cv::IMREAD_GRAYSCALE);
    // std::cout << "c:" << distorted_img.channels() << std::endl;

    cv::Mat undistorted_img(distorted_img.rows, distorted_img.cols, distorted_img.type());
    // cv::imshow("distorted", distorted_img);
    // cv::waitKey(0);
    for (int v = 0; v < distorted_img.rows; ++v) {
        for (int u = 0; u < distorted_img.cols; ++u) {
            // caculate origin x,y
            // u = x*fx+cx, v = y*fy+cy
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_corrected = x * (1 + k1 * pow(r, 2) + k2 * pow(r, 4)) +
                2 * p1 * x * y + p2 * (pow(r, 2) + 2 * pow(x, 2));
            double y_corrected = y * (1 + k1 * pow(r, 2) + k2 * pow(r, 4)) +
                2 * p2 * x * y + p1 * (pow(r, 2) + 2 * pow(y, 2));
            double u_corrected = x_corrected * fx + cx;
            double v_corrected = y_corrected * fy + cy;

            if (u_corrected >= 0 && u_corrected < distorted_img.cols &&
                v_corrected >= 0 && v_corrected < distorted_img.rows) {
                undistorted_img.at<uchar >(v, u) = distorted_img.at<uchar >((int)v_corrected, (int)u_corrected);
            } else {
                undistorted_img.at<uchar >(v, u) = 0;
            }
        }
    }
    cv::imshow("distort", distorted_img);
    cv::imshow("undistort", undistorted_img);

    // use opencv api
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    std::cout << "camera K\n:" << camera_matrix << std::endl;
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);
    cv::Mat dst;
    cv::undistort(distorted_img, dst, camera_matrix, dist_coeffs);

    cv::imshow("opencv undistort", dst);
    cv::waitKey(0);
    return 0;
}