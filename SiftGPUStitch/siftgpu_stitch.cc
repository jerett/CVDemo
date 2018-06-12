//
// Created by jerett on 18-6-11.
//

#include <iostream>
#include <vector>
#include <SiftGPU/SiftGPU.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

Size CaculateWarpSize(const Mat &corners, const Mat &H) {
    Mat warp_corners = H * corners;
    warp_corners.col(0) /= warp_corners.at<double>(2, 0);
    warp_corners.col(1) /= warp_corners.at<double>(2, 1);
    warp_corners.col(2) /= warp_corners.at<double>(2, 2);
    warp_corners.col(3) /= warp_corners.at<double>(2, 3);

    cv::Mat all_corners;
    hconcat(corners, warp_corners, all_corners);
    CV_LOG_INFO(NULL, "all coerners:\n" << all_corners);
//    cv::Mat corner_min;
//    cv::reduce(warp_corners, corner_min, 1, CV_REDUCE_MIN);
//    CV_LOG_INFO(NULL, "corner min:\n" << corner_min);
    cv::Mat corner_max;
    cv::reduce(all_corners, corner_max, 1, CV_REDUCE_MAX);
    CV_LOG_INFO(NULL, "corner max:\n" << corner_max);
//    return Size(corner_max.at<double>(0, 0) - corner_min.at<double>(0, 0),
//                corner_max.at<double>(1, 0) - corner_min.at<double>(1, 0));
//    return Size(corner_max.at<double>(0, 0) ,
//                corner_max.at<double>(1, 0) - corner_min.at<double>(1, 0));
//    return Size(corner_max.at<double>(0, 0), corner_max.at<double>(1, 0));
    return Size(corner_max.at<double>(0, 0), corner_max.at<double>(1, 0));
}

cv::Mat TestFindHomography(const Mat &src1, const Mat &src2,
                           const std::vector<Point2f> &src1_keypoints, const std::vector<Point2f> &src2_keypoints) {
    Mat H = findHomography(src2_keypoints, src1_keypoints, CV_RANSAC);
    CV_LOG_INFO(NULL, "estimate H:\n" << H << "\n size:" << H.size);
    double corners_data[4][3] = {
            {0,         0,         1},
            {0,         src2.rows, 1},
            {src2.cols, 0,         1},
            {src2.cols, src2.rows, 1},
    };
    Mat corners = Mat(4, 3, CV_64FC1, corners_data).t();
    CV_LOG_INFO(NULL, "corners:\n" << corners);
    auto warp_size = CaculateWarpSize(corners, H);
    CV_LOG_INFO(NULL, "size:\n" << warp_size);
    cv::Mat out;
    warpPerspective(src2, out, H, warp_size);
    return out;
}


int main(int argc, char *argv[]) {
    const char *file0 = "uttower1.jpg";
    Mat mat0 = imread(file0, CV_LOAD_IMAGE_COLOR);

    const char *file1 = "uttower2.jpg";
    Mat mat1 = imread(file1, CV_LOAD_IMAGE_COLOR);

    SiftGPU sift;
    int support = sift.CreateContextGL();
    if (support != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
        std::cerr << "SiftGPU is not supported" << std::endl;
        return 2;
    }
    vector<SiftGPU::SiftKeypoint> src0_keys;
    vector<float> src0_descriptors;

    vector<SiftGPU::SiftKeypoint> src1_keys;
    vector<float> src1_descriptors;

    {
        sift.RunSIFT(file0);

        // 获取关键点与描述子
        int num = sift.GetFeatureNum();
        src0_descriptors.resize(128 * num);
        src0_keys.resize(num);
        CV_LOG_INFO(NULL, "src0 feature num:" << num);
        sift.GetFeatureVector(&src0_keys[0], &src0_descriptors[0]);

        sift.RunSIFT(file1);
        // 获取关键点与描述子
        num = sift.GetFeatureNum();
        src1_descriptors.resize(128 * num);
        src1_keys.resize(num);
        CV_LOG_INFO(NULL, "src1 feature num:" << num);
        sift.GetFeatureVector(&src1_keys[0], &src1_descriptors[0]);
//
//        cv::Mat circleMat;
//        cv::drawKeypoints(mat0, kpts, circleMat);
//        cv::imshow("GPUSift", circleMat);
    }

    {
        SiftMatchGPU matcher;
        CV_Assert(matcher.VerifyContextGL() != 0);
        matcher.SetDescriptors(0, src0_keys.size(), &src0_descriptors[0]);
        matcher.SetDescriptors(1, src1_keys.size(), &src1_descriptors[0]);
        int match_buf[4096][2];
        int nmatch = matcher.GetSiftMatch(4096, match_buf);
        CV_LOG_INFO(NULL, "match num:" << nmatch);


        std::vector<KeyPoint> src0_cv_kpts;
        std::vector<KeyPoint> src1_cv_kpts;
        for (auto &skt : src0_keys) {
            KeyPoint point(skt.x, skt.y, skt.s);
            src0_cv_kpts.push_back(point);
        }
        for (auto &skt : src1_keys) {
            KeyPoint point(skt.x, skt.y, skt.s);
            src1_cv_kpts.push_back(point);
        }
        std::vector<DMatch> matches;
        for (int i = 0; i < nmatch; ++i) {
            DMatch match;
            match.queryIdx = match_buf[i][0];
            match.trainIdx = match_buf[i][1];
            matches.push_back(match);
        }
        cv::Mat draw_image;
        cv::drawMatches(mat0, src0_cv_kpts,
                        mat1, src1_cv_kpts,
                        matches, draw_image);
//        CV_LOG_INFO(NULL, "input1 desc size:" << src1.descriptors.rows << " matches size:" << good_matches.size());
        cv::imshow("SiftGPUMatch", draw_image);

        std::vector<Point2f> src0_good_keypoints;
        std::vector<Point2f> src1_good_keypoints;
        for (const DMatch &good_match : matches) {
            auto &p1 = src0_cv_kpts[good_match.queryIdx].pt;
            auto &p2 = src1_cv_kpts[good_match.trainIdx].pt;
            src0_good_keypoints.push_back(p1);
            src1_good_keypoints.push_back(p2);
        }
        cv::Mat warp_image = TestFindHomography(mat0, mat1, src0_good_keypoints, src1_good_keypoints);
        imshow("warp_src2", warp_image);
        mat0.copyTo(warp_image(cv::Rect(0, 0, mat0.cols, mat0.rows)));
        imshow("SiftGPU stitch", warp_image);
        imwrite("siftgpu_stitch.jpg", warp_image);
    }

//    {
//        auto detector = xfeatures2d::SIFT::create();
//        cv::Mat descriptors;
//        std::vector<KeyPoint> kpts;
//        auto startClock = cv::getTickCount();
//        detector->detectAndCompute(mat, noArray(), kpts, descriptors);
//        auto pass = (cv::getTickCount() - startClock) / cv::getTickFrequency();
//        std::cout << "sift cpu pass:" << pass << std::endl;
//        std::cout <<  kpts.size() << " points detected" << pass << std::endl;
//
//        cv::Mat circleMat;
//        cv::drawKeypoints(mat, kpts, circleMat);
//
//        cv::imshow("CPUSift", circleMat);
//    }
//
//    {
//        auto detector = xfeatures2d::SURF::create();
//        cv::Mat descriptors;
//        std::vector<KeyPoint> kpts;
//        detector->detectAndCompute(mat, noArray(), kpts, descriptors);
//
//        auto startClock = cv::getTickCount();
//        detector->detectAndCompute(mat, noArray(), kpts, descriptors);
//        auto pass = (cv::getTickCount() - startClock) / cv::getTickFrequency();
//        std::cout << "surf cpu pass:" << pass << std::endl;
//        std::cout <<  kpts.size() << " points detected" << pass << std::endl;
//
//        cv::Mat circleMat;
//        cv::drawKeypoints(mat, kpts, circleMat);
//        cv::imshow("CPUSurf", circleMat);
//    }

    cv::waitKey(0);
    return 0;
}