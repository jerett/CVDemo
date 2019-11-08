//
// Created by jerett on 2019/11/7.
//

#include <iostream>
#include <fstream>
#include <thread>
#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

const std::string color_dir("color/");
const std::string depth_dir("depth/");

typedef Eigen::Matrix<double, 6, 1> Vector6d;
void showPointCloud(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char *argv[]) {
    std::vector<cv::Mat> color_imgs;
    std::vector<cv::Mat> depth_imgs;
    std::vector<Sophus::SE3d> poses;
    // std::cout << "sophus " << std::endl;

    std::ifstream pose_txt("pose.txt");
    for (int i = 0; i < 5; ++i) {
        std::string color_img_path = color_dir + std::to_string(i + 1) + ".png";
        std::string depth_img_path = depth_dir + std::to_string(i + 1) + ".pgm";

        auto color_img = cv::imread(color_img_path);
        auto depth_img = cv::imread(depth_img_path, -1);
        color_imgs.push_back(color_img);
        depth_imgs.push_back(depth_img);

        double pose[7];
        for (int j = 0; j < 7; ++j) {
            pose_txt >> pose[j];
        }
        Sophus::SE3d pose3d(Eigen::Quaterniond(pose[6], pose[3], pose[4], pose[5]),
                            Eigen::Vector3d(pose[0], pose[1], pose[2]));
        poses.push_back(pose3d);
        std::cout << std::endl;
    }

    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depth_scale = 1000.0;
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < 5; ++i) {
        auto img = color_imgs[i];
        auto depth_img = depth_imgs[i];

        auto T = poses[i];
        for (int v = 0; v < img.rows; ++v) {
            for (int u = 0; u < img.cols; ++u) {
                double depth = depth_img.at<unsigned short>(v, u) / depth_scale;
                if (depth <= 0) continue;
                double x = (u - cx) / fx * depth;
                double y = (v - cy) / fy * depth;
                double z = depth;
                Eigen::Vector3d p(x, y, z);
                Eigen::Vector3d pw = T * p;
                // std::cout << "pw:" << pw << std::endl;

                Vector6d point;
                point[0] = pw[0];
                point[1] = pw[1];
                point[2] = pw[2];
                // order in r, g, b
                point[5] = img.at<cv::Vec3b>(v, u)[0] / 255.0;
                point[4] = img.at<cv::Vec3b>(v, u)[1] / 255.0;
                point[3] = img.at<cv::Vec3b>(v, u)[2] / 255.0;
                pointcloud.push_back(point);
            }
        }
    }

    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {
    if (pointcloud.empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[4], p[5]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}