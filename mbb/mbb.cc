//
// Created by jerett on 2019/9/25.
//
#include <iostream>
#include <opencv2/opencv.hpp>

// test image blender
int main(int argc, char *argv[]) {
    cv::Mat img1 = cv::imread("apple.jpg");
    cv::Mat img2 = cv::imread("orange.jpg");
    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
    // cv::waitKey(0);

    img1.convertTo(img1, CV_16S);
    img2.convertTo(img2, CV_16S);

    cv::Mat mask1(img1.size(), CV_8U);
    mask1(cv::Rect(0, 0, img1.cols / 2, img1.rows)).setTo(255);
    mask1(cv::Rect(img1.cols / 2, 0, img1.cols - img1.cols / 2, img1.rows)).setTo(0);

    cv::Mat mask2(img2.size(), CV_8U);
    mask2(cv::Rect(0, 0, img2.cols / 2, img2.rows)).setTo(0);
    mask2(cv::Rect(img2.cols / 2, 0, img2.cols - img2.cols / 2, img2.rows)).setTo(255);

    cv::detail::MultiBandBlender blender(false);
    cv::Rect roi_dst(0, 0, std::max(img1.cols, img2.cols), std::max(img1.rows, img2.rows));
    blender.prepare(roi_dst);
    blender.feed(img1, mask1, cv::Point(0, 0));
    blender.feed(img2, mask2, cv::Point(0, 0));

    cv::Mat multiband_dst;
    cv::Mat mask_final;
    blender.blend(multiband_dst, mask_final);

    multiband_dst.convertTo(multiband_dst, CV_8U);

    // cv::imshow("blend", mask_final);
    cv::imshow("blend", multiband_dst);
    cv::waitKey();
    return 0;
}