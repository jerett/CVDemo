//
// Created by jerett on 2019/11/11.
//

#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <fstream>

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

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
        (
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        );
}

using VecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

void pose_estimation_3d3d(
    const std::vector<Point3f> &pts1,
    const std::vector<Point3f> &pts2,
    Mat &R, Mat &t) {
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    std::vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    std::cout << "W=" << W << std::endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    std::cout << "U=" << U << std::endl;
    std::cout << "V=" << V << std::endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
                            R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

int main(int argc, char *argv[]) {
    auto img1 = cv::imread("1.png");
    auto img1_depth = cv::imread("1_depth.png", cv::IMREAD_UNCHANGED);
    auto img2 = cv::imread("2.png");
    auto img2_depth = cv::imread("2_depth.png", cv::IMREAD_UNCHANGED);

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;
    match(img1, img2, keypoints_1, keypoints_2, matches);

    std::vector<Point3f> img1_kp_3d;
    std::vector<Point3f> img2_kp_3d;
    for (const auto &match : matches) {
        const auto &kp1 = keypoints_1[match.queryIdx];
        ushort d1 = img1_depth.at<ushort>(static_cast<int>(kp1.pt.y), static_cast<int>(kp1.pt.x));
        const auto &kp2 = keypoints_2[match.trainIdx];
        ushort d2 = img2_depth.at<ushort>(static_cast<int>(kp2.pt.y), static_cast<int>(kp2.pt.x));
        if (d1 == 0 || d2 == 0) continue;
        float depth1 = d1 / 5000.0;
        float depth2 = d2 / 5000.0;
        auto pt1_cam = pixel2cam(kp1.pt, K);
        img1_kp_3d.push_back(Point3f(pt1_cam.x * depth1, pt1_cam.y * depth1, depth1));
        auto pt2_cam = pixel2cam(kp2.pt, K);
        img2_kp_3d.push_back(Point3f(pt2_cam.x * depth2, pt2_cam.y * depth2, depth2));
    }
    std::cout << "3d-3d pairs:" << img1_kp_3d.size() << std::endl;

    cv::Mat R, t;
    pose_estimation_3d3d(img1_kp_3d, img2_kp_3d, R, t);
    std::cout << "R:\n" << R << std::endl << "t:\n" << t << std::endl;
    return 0;
}



