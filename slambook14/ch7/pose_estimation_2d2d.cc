//
// Created by jerett on 2019/11/11.
//

#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;

// 相机内参,TUM Freiburg2
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

void match(const Mat &img1,
           const Mat &img2,
           std::vector<KeyPoint> &keypoints_1,
           std::vector<KeyPoint> &keypoints_2,
           std::vector<DMatch> &matches) {
    cv::Mat img1_desc, img2_desc;

    auto orb_feature = cv::ORB::create();
    orb_feature->detect(img1, keypoints_1);
    orb_feature->compute(img1, keypoints_1, img1_desc);
    orb_feature->detect(img2, keypoints_2);
    orb_feature->compute(img2, keypoints_2, img2_desc);

    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    matcher->match(img1_desc, img2_desc, matches);

    auto min_max = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) {
        return m1.distance < m2.distance;
    });
    std::cout << "min distance:" << min_max.first->distance << std::endl;
    std::cout << "max distance:" << min_max.second->distance << std::endl;

    std::vector<cv::DMatch> good_matches;
    for (const auto &match : matches) {
        if (match.distance <= std::max(2 * min_max.first->distance, 30.0f)) {
            good_matches.push_back(match);
        }
    }
    std::cout << "find " << good_matches.size() << " matches" << std::endl;
    matches = good_matches;
}

void pose_estimation_2d2d(
    std::vector<KeyPoint> keypoints_1,
    std::vector<KeyPoint> keypoints_2,
    std::vector<DMatch> matches,
    Mat &R, Mat &t) {
    std::vector<Point2f> points1;
    std::vector<Point2f> points2;

    for (const auto &match : matches) {
        points1.push_back(keypoints_1[match.queryIdx].pt);
        points2.push_back(keypoints_2[match.trainIdx].pt);
    }

    // F = K(-T) * E * K(-1)
    Mat F = findFundamentalMat(points1, points2, FM_8POINT);
    std::cout << "F:\n" << F << std::endl;

    Mat E = findEssentialMat(points1, points2, K, FM_RANSAC);
    std::cout << "E:\n" << E << std::endl;

    Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
    double focal_length = 521;      //相机焦距, TUM dataset标定值
    Mat E2 = findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout << "E2:\n" << E2 << std::endl;

    recoverPose(E, points1, points2, K, R, t);
    std::cout << "R:\n" << R << std::endl << "t:\n" << t << std::endl;
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
        (
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        );
}

int main(int argc, char *argv[]) {
    auto img1 = cv::imread("1.png");
    auto img2 = cv::imread("2.png");

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;
    match(img1, img2, keypoints_1, keypoints_2, matches);

    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //-- 验证E=t^R*scale
    Mat t_x =
        (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
            t.at<double>(2, 0), 0, -t.at<double>(0, 0),
            -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    std::cout << "t^*R:\n" << t_x * R << std::endl;

    //
    for (const auto &match : matches) {
        const auto &uv1 = keypoints_1[match.queryIdx].pt;
        const auto &uv2 = keypoints_2[match.trainIdx].pt;
        const auto pt1 = pixel2cam(uv1, K);
        const Mat pt1_3d = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        const auto pt2 = pixel2cam(uv2, K);
        const Mat pt2_3d = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

        Mat d = pt2_3d.t() * t_x * R * pt1_3d;
        std::cout << " epipolar constraint = " << d << std::endl;
    }
    return 0;
}