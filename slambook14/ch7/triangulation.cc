//
// Created by jerett on 2019/11/11.
//

#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;

// 相机内参,TUM Freiburg2
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
        (
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        );
}

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

void triangulation(const std::vector<KeyPoint> &keypoints_1,
                   const std::vector<KeyPoint> &keypoints_2,
                   const std::vector<DMatch> &matches,
                   const Mat &R, const Mat &t,
                   std::vector<Point3d> &points) {
    std::vector<Point2f> points1;
    std::vector<Point2f> points2;

    for (const auto &match : matches) {
        points1.push_back(pixel2cam(keypoints_1[match.queryIdx].pt, K));
        points2.push_back(pixel2cam(keypoints_2[match.trainIdx].pt, K));
    }
    Mat T1 = (Mat_<double>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    Mat T2 = (Mat_<double>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    // cv::triangulatePoints()
    // 4 * N
    cv::Mat points4d_array;
    triangulatePoints(T1, T2, points1, points2, points4d_array);
    for (int i = 0; i < points4d_array.cols; ++i) {
        Mat point4d = points4d_array.col(i);
        point4d /= point4d.at<float>(3, 0);
        Point3d point3d(point4d.at<float>(0, 0),
                        point4d.at<float>(1, 0),
                        point4d.at<float>(2, 0));
        points.push_back(point3d);
    }
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
        (Mat_<double>(3, 3) <<
            0, -t.at<double>(2, 0), t.at<double>(1, 0),
            t.at<double>(2, 0), 0, -t.at<double>(0, 0),
            -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    std::cout << "t^*R:\n" << t_x * R << std::endl;

    // 测试对齐约束
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

    std::vector<Point3d> points3d;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points3d);

    // 测试重投影误差
    for (int i = 0; i < matches.size(); ++i) {
        const auto &point3d = points3d[i];
        auto match = matches[i];
        const double depth = point3d.z;

        const auto &uv1 = keypoints_1[match.queryIdx].pt;
        const auto pt1 = pixel2cam(uv1, K);

        const Point2d pt1_pro(point3d.x / point3d.z, point3d.y / point3d.z);
        std::cout << "depth:" << depth << std::endl;
        std::cout << "pt1 3d:" << pt1 << " pt1 3d pro:" << pt1_pro << std::endl;

        const auto &uv2 = keypoints_2[match.trainIdx].pt;
        const auto pt2 = pixel2cam(uv2, K);
        Mat pt2_3d_pro = R * (Mat_<double>(3, 1) << point3d.x, point3d.y, point3d.z) + t;
        pt2_3d_pro /= pt2_3d_pro.at<double>(2, 0);

        std::cout << "pt2 3d:" << pt2 << " pt2 3d pro:" << pt2_3d_pro.t() << std::endl;
    }
    return 0;
}