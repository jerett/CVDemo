//
// Created by jerett on 2019/11/11.
//

#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

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

void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose) {
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    const int iterations = 10;
    // const int iterations = 1;
    double cost = 0;
    double last_cost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    using Vector6d = Eigen::Matrix<double, 6, 1>;
    for (int iter = 0; iter < iterations; ++iter) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        for (int i = 0; i < points_3d.size(); ++i) {
            const Eigen::Vector3d &pc = pose * points_3d[i];
            double x = fx * pc[0] / pc[2] + cx;
            double y = fy * pc[1] / pc[2] + cy;
            const Eigen::Vector2d proj(x, y);
            const Eigen::Vector2d e1 = proj - points_2d[i];
            const Eigen::Vector2d e2 = points_2d[i] - proj;
            assert(e1.squaredNorm() == e2.squaredNorm());
            cost += e2.squaredNorm();

            Eigen::Matrix<double, 2, 6> J;
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            J << -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e2;
        }
        Vector6d dx;
        dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            std::cout << "dx is nan" << std::endl;
            break;
        }
        pose = Sophus::SE3d::exp(dx) * pose;
        last_cost = cost;
        std::cout << "iter:" << iter << ", cost:" << cost << std::endl;
        if (dx.norm() < 1e-6) {
            std::cout << "converge at iter:" << iter << std::endl;
            break;
        }
    }
}

int main(int argc, char *argv[]) {
    auto img1 = cv::imread("1.png");
    auto img1_depth = cv::imread("1_depth.png", cv::IMREAD_UNCHANGED);
    auto img2 = cv::imread("2.png");

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;
    match(img1, img2, keypoints_1, keypoints_2, matches);

    std::vector<Point3f> img1_kp_3d;
    std::vector<Point2f> img2_kp;
    for (const auto &match : matches) {
        const auto &kp = keypoints_1[match.queryIdx];
        ushort d = img1_depth.at<ushort >(static_cast<int>(kp.pt.y), static_cast<int>(kp.pt.x));
        // std::cout << "d:" << d << std::endl;
        if (d == 0) continue;
        float depth = d / 5000.0;
        auto pt1_cam = pixel2cam(kp.pt, K);
        img1_kp_3d.push_back(Point3f(pt1_cam.x * depth, pt1_cam.y * depth, depth));
        img2_kp.push_back(keypoints_2[match.trainIdx].pt);
    }
    std::cout << "3d-2d pairs:" << img1_kp_3d.size() << std::endl;

    Mat r_vec, t;
    cv::solvePnP(img1_kp_3d, img2_kp, K, Mat(), r_vec, t);
    Mat R;
    cv::Rodrigues(r_vec, R);
    std::cout << "R:\n" << R << std::endl;
    std::cout << "t:\n" << t << std::endl;

    VecVector3d img1_pts_3d_eigen;
    VecVector2d img2_pts_2d_eigen;
    for (int i = 0; i < img1_kp_3d.size(); ++i) {
        const auto &kp_3d = img1_kp_3d[i];
        const auto &kp_2d = img2_kp[i];
        img1_pts_3d_eigen.push_back((Eigen::Vector3d(kp_3d.x, kp_3d.y, kp_3d.z)));
        img2_pts_2d_eigen.push_back(Eigen::Vector2d(kp_2d.x, kp_2d.y));
    }
    Sophus::SE3d pose_gn;
    std::cout << "init pose_gn:" << pose_gn.matrix() << std::endl;
    bundleAdjustmentGaussNewton(img1_pts_3d_eigen, img2_pts_2d_eigen, K, pose_gn);
    std::cout << "pose by gn:\n" << pose_gn.matrix() << std::endl;
    return 0;
}



