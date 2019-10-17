//
// Created by jerett on 2019/10/15.
//


#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat DrawHist(const int hist[256]) {
    int max_val = hist[255];

    int ybin = 256;
    Mat hist_img(ybin, 256, CV_8U, Scalar(0));
    for (int i = 0; i < 256; i++) {
        float binValue = hist[i];
        line(hist_img, Point(i, ybin - 1), Point(i, (max_val - binValue) * ybin / max_val), Scalar(255));
    }
    return hist_img;
}

void Hist(const Mat &img, int hist[256]) {
    // pdf
    for (int y = 0; y < img.rows; ++y) {
        const uchar *p = img.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            const uchar r = p[x];
            const uchar g = p[x + 1];
            const uchar b = p[x + 2];
            ++hist[r];
            ++hist[g];
            ++hist[b];
        }
    }

    // cdf
    for (int i = 1; i < 256; ++i) {
        hist[i] = hist[i] + hist[i - 1];
    }
}

void MatchHist(const int src_hist[256], const int target_hist[256], int map[256]) {
    int j = 0;
    for (int i = 0; i < 256; ++i) {
        for (; j < 256; ++j) {
            if (target_hist[j] >= src_hist[i]) {
                map[i] = j;
                break;
            }
        }
    }
}

void MapImage(const Mat &src, const Mat &target, const int map[256], Mat &dst) {
    dst = Mat(src.size(), src.type());
    int size = src.rows * src.cols;
    const uchar *src_data = src.data;
    uchar *dst_data = dst.data;

    // for (int i = 0; i < size; ++i) {
    //     const uchar b = src_data[3 * i];
    //     const uchar g = src_data[3 * i + 1];
    //     const uchar r = src_data[3 * i + 2];
    //
    //     float gray = (r + g + b) / 3.0f;
    //     float k = map[int(gray)] / (gray + 0.01f);
    //     dst_data[3 * i] = std::min(int(b * k), 255);
    //     dst_data[3 * i + 1] = std::min(int(g * k), 255);
    //     dst_data[3 * i + 2] = std::min(int(r * k), 255);
    // }

    for (int i = 0; i < size * 3; ++i) {
        dst_data[i] = map[src_data[i]];
    }

}

int main(int argc, char *argv[]) {
    // std::cout << "hist match.." << std::endl;
    cv::Mat src_img = imread("DSC02984.JPG");
    int src_hist[256] = {};
    Hist(src_img, src_hist);

    cv::Mat target_img = imread("DSC02986.JPG");
    // cv::Mat target_img = imread("over_ev.jpg");
    int target_hist[256] = {};
    Hist(target_img, target_hist);

    auto src_hist_img = DrawHist(src_hist);
    auto target_hist_img = DrawHist(target_hist);

    int map[256];
    MatchHist(src_hist, target_hist, map);
    auto map_img = DrawHist(map);

    Mat dst;
    MapImage(src_img, target_img, map, dst);
    int dst_hist[256];
    Hist(dst, dst_hist);
    auto dst_hist_img = DrawHist(target_hist);

    imshow("src hist", src_hist_img);
    imshow("target hist", target_hist_img);
    imshow("map hist", map_img);
    imshow("dst hist", dst_hist_img);
    imshow("dst img", dst);
    waitKey(0);
    return 0;
}

