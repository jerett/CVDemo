//
// Created by jerett on 18-6-6.
//

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

struct InputImage {
public:
    bool Load(const std::string &path) {
        this->path = path;
        image = cv::imread(path);
        return image.rows > 0 && image.cols > 0;
    }

    void DetectAndCompute(Feature2D &featureDetector) {
        featureDetector.detectAndCompute(image, noArray(), keypoints, descriptors);
        CV_LOG_INFO(NULL, path << " " << keypoints.size() << " keypoints detected.");
    }

    Mat image;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    std::string path;
};

void GetMatch(InputImage &src1, InputImage &src2, std::vector<DMatch> &good_matches) {
//    auto featureDetector = xfeatures2d::SURF::create();
    auto featureDetector = xfeatures2d::SIFT::create();
    src1.DetectAndCompute(*featureDetector);
    src2.DetectAndCompute(*featureDetector);

    // match
    auto matcher = BFMatcher::create(NORM_L2, false);
    std::vector<std::vector<DMatch>> matches;
//    matcher->match(inputImage1.descriptors, inputImage2.descriptors, matches);
//    matcher->match(src1.descriptors, src2.descriptors, matches);
    matcher->knnMatch(src1.descriptors, src2.descriptors, matches, 2);
    for (int i = 0; i < matches.size(); ++i) {
        if (matches[i][0].distance <= matches[i][1].distance * 0.6) {
            good_matches.push_back(matches[i][0]);
        }
    }
}

Size GetWarpSize(std::vector<InputImage> &imgs, const std::vector<Mat> &H_array) {
    Mat all_corners;
    std::vector<Mat> corners_array;
    {
        for (int i = 0; i < imgs.size(); ++i) {
            const InputImage &src = imgs[i];
            const cv::Mat &H = H_array[i];

            double corners_data[4][3] = {
                    {0,              0,              1},
                    {0,              src.image.rows, 1},
                    {src.image.cols, 0,              1},
                    {src.image.cols, src.image.rows, 1},
            };
            Mat corners = Mat(4, 3, CV_64FC1, corners_data).t();
            Mat warp_corners = H * corners;
            warp_corners.col(0) /= warp_corners.at<double>(2, 0);
            warp_corners.col(1) /= warp_corners.at<double>(2, 1);
            warp_corners.col(2) /= warp_corners.at<double>(2, 2);
            warp_corners.col(3) /= warp_corners.at<double>(2, 3);
            corners_array.push_back(corners);
            corners_array.push_back(warp_corners);
//            hconcat(corners, all_corners, all_corners);
//            hconcat(warp_corners, all_corners, all_corners);
//            all_corners.push_back(corners);
//            all_corners.push_back(warp_corners);
        }
    }
    hconcat(&corners_array[0], corners_array.size(), all_corners);
//    cv::Mat all_corners;
//    hconcat(corners, warp_corners, all_corners);
    CV_LOG_INFO(NULL, "all coerners:\n" << all_corners);
//    cv::Mat corner_min;
//    cv::reduce(warp_corners, corner_min, 1, CV_REDUCE_MIN);
//    CV_LOG_INFO(NULL, "corner min:\n" << corner_min);
    cv::Mat corner_max;
    cv::reduce(all_corners, corner_max, 1, CV_REDUCE_MAX);
    CV_LOG_INFO(NULL, "corner max:\n" << corner_max);
    return Size(corner_max.at<double>(0, 0), corner_max.at<double>(1, 0));
}

cv::Mat Warp(const InputImage &src, const Mat &H, const Size warp_size) {
    cv::Mat warp;
    warpPerspective(src.image, warp, H, warp_size);
    return warp;
}

cv::Mat GetH(const std::vector<Point2f> &src1_keypoints, const std::vector<Point2f> &src2_keypoints) {
    Mat H = findHomography(src2_keypoints, src1_keypoints, CV_RANSAC);
    CV_LOG_INFO(NULL, "estimate H:\n" << H << "\n size:" << H.size);
    return H;
}

cv::Mat GetH2(const std::vector<Point2f> &src1_keypoints, const std::vector<Point2f> &src2_keypoints) {
    Mat R = estimateRigidTransform(src2_keypoints, src1_keypoints, true);
    Mat H(3, 3, R.type());
    R.copyTo(H(Rect(0, 0, 3, 2)));
    H.at<double>(2, 0) = 0;
    H.at<double>(2, 1) = 0;
    H.at<double>(2, 2) = 1;
    return H;
}

int main(int argc, char *argv[]) {
    std::vector<InputImage> imgs(argc - 1);
//    InputImage src1;
//    InputImage src2;
//    CV_Assert(imgs[0].Load("uttower1.jpg"));
//    CV_Assert(imgs[1].Load("uttower2.jpg"));
//    CV_Assert(src1.Load("Image1.jpg"));
//    CV_Assert(src2.Load("Image2.jpg"));
//    CV_Assert(imgs[0].Load("yosemite1.jpg"));
//    CV_Assert(imgs[1].Load("yosemite2.jpg"));
//    CV_Assert(imgs[2].Load("yosemite3.jpg"));

    for (int i = 0; i< argc-1; ++i) {
        imgs[i].Load(argv[i+1]);
    }

    // H array corresponding to each img, img[0] is ID matrix.
    std::vector<Mat> H_array;
    cv::Mat eyeH = Mat::eye(Size(3, 3), CV_64FC1);
    H_array.push_back(eyeH);
    for (int i = 0; i < imgs.size() - 1; ++i) {
        InputImage &left = imgs[i];
        InputImage &right = imgs[i + 1];

        std::vector<DMatch> good_matches;
        GetMatch(left, right, good_matches);
        {
            cv::Mat draw_image;
            cv::drawMatches(left.image, left.keypoints,
                            right.image, right.keypoints,
                            good_matches, draw_image);
            CV_LOG_INFO(NULL,
                        left.path << "," << right.path << (i + 1) << " good matches size:" << good_matches.size());
            cv::imshow("BFMatch", draw_image);
        }
        // find H
        std::vector<Point2f> src1_good_keypoints;
        std::vector<Point2f> src2_good_keypoints;
        for (const DMatch &good_match : good_matches) {
            auto &p1 = left.keypoints[good_match.queryIdx].pt;
            auto &p2 = right.keypoints[good_match.trainIdx].pt;
            src1_good_keypoints.push_back(p1);
            src2_good_keypoints.push_back(p2);
        }
        Mat H = GetH(src1_good_keypoints, src2_good_keypoints);
        H = H * H_array[i];
        H_array.push_back(H);
    }

    // get stitch out size
    Size out_size = GetWarpSize(imgs, H_array);

    // warp each img, and get overlap mask
    std::vector<Mat> warp_imgs;
    cv::Mat overlap;
    for (int i = 0; i < imgs.size(); ++i) {
        const InputImage &img = imgs[i];
        const Mat &H = H_array[i];
        Mat warp_img = Warp(img, H, out_size);
        imshow("warp " + img.path, warp_img);

        Mat mask = (warp_img != 0);
        mask /= 255;
        mask.convertTo(mask, CV_32FC3);
        warp_img.convertTo(warp_img, CV_32FC3);
        warp_imgs.push_back(warp_img);

        if (i == 0) overlap = mask;
        else overlap += mask;
    }
    overlap.setTo(1, (overlap == 0));
//    CV_LOG_INFO(NULL, overlap);
    // add warp img
    cv::Mat stitch;
    for (int i = 0; i < warp_imgs.size(); ++i) {
        if (i == 0) stitch = warp_imgs[i];
        else stitch += warp_imgs[i];
    }
    stitch /= overlap;
    stitch.convertTo(stitch, CV_8UC3);
    imshow("stitch", stitch);
    imwrite("stitch.jpg", stitch);

    cv::waitKey(0);
    return 0;
}