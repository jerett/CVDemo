//
// Created by jerett on 18-6-6.
//

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

bool visual_dbg = false;
std::string blend_method = "mbb";
std::vector<std::string> image_names;

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

static void printUsage() {
    cout <<
         "Rotation model images stitcher.\n\n"
         "Stitch img1 img2 [...imgN] [flags]\n\n"
         "Flags:\n"
         "  --blender(no|mbb)\n"
         "      Set blender method, The default value is 'mbb'.\n"
         "  --visual_dbg\n"
         "      Visualize debug.\n";
}

static int parseCmdArgs(int argc, char** argv) {
    if (argc == 1) {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?") {
            printUsage();
            return -1;
        } else if (string(argv[i]) == "--visual_dbg") {
            visual_dbg = true;
        } else if (string(argv[i]) == "--blender") {
            i++;
            blend_method = string(argv[i]);
        } else {
            image_names.push_back(string(argv[i]));
        }
    }
    return 0;
}

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
                    {0,                                    0,                                    1},
                    {0,                                    static_cast<double >(src.image.rows), 1},
                    {static_cast<double >(src.image.cols), 0,                                    1},
                    {static_cast<double >(src.image.cols), static_cast<double >(src.image.rows), 1},
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
    cv::reduce(all_corners, corner_max, 1, cv::REDUCE_MAX);
    CV_LOG_INFO(NULL, "corner max:\n" << corner_max);
    return Size(corner_max.at<double>(0, 0), corner_max.at<double>(1, 0));
}

cv::Mat GetH(const std::vector<Point2f> &src1_keypoints, const std::vector<Point2f> &src2_keypoints) {
    Mat H = findHomography(src2_keypoints, src1_keypoints, cv::RANSAC);
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

cv::Mat StitchWithNoBlend(const std::vector<Mat> &warp_imgs, const std::vector<Mat> &warp_masks) {
    // add warp img
    cv::Mat stitch(warp_imgs[0].size(), CV_32FC(warp_imgs[0].channels()));
    cv::Mat stitch_mask(warp_masks[0].size(), CV_8U);
    stitch_mask.setTo(0);
    for (int i = 0; i < warp_imgs.size(); ++i) {

        cv::Mat warp_img_f;
        warp_imgs[i].convertTo(warp_img_f, stitch.type());
        add(warp_img_f, stitch, stitch, warp_masks[i]);
    }
    stitch.convertTo(stitch, CV_8UC3);
    return stitch;
}

cv::Mat StitchWithMbbBlend(const std::vector<Mat> &warp_imgs, const std::vector<Mat> &warp_masks) {
    cv::Rect roi_dst(0, 0, warp_imgs[0].cols, warp_imgs[0].rows);
    cv::detail::MultiBandBlender blender(false);
    blender.prepare(roi_dst);

    for (int i = 0; i < warp_imgs.size(); ++i) {
        blender.feed(warp_imgs[i], warp_masks[i], cv::Point(0, 0));
    }
    cv::Mat stitch;
    cv::Mat stitch_mask;
    blender.blend(stitch, stitch_mask);
    stitch.convertTo(stitch, CV_8U);
    return stitch;
}


int main(int argc, char *argv[]) {
    int retval = parseCmdArgs(argc, argv);
    if (retval != 0) return retval;

    std::vector<InputImage> imgs(image_names.size());
    for (int i = 0; i < image_names.size(); ++i) {
        imgs[i].Load(image_names[i]);
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
            // cv::imshow("BFMatch", draw_image);
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
    std::vector<Mat> warp_masks;
    cv::Mat overlap;
    for (int i = 0; i < imgs.size(); ++i) {
        const InputImage &img = imgs[i];
        const Mat mask = cv::Mat(img.image.size(), CV_8UC1, cv::Scalar(255));

        const Mat &H = H_array[i];
        Mat warp_img, warp_mask;
        warpPerspective(img.image, warp_img, H, out_size);
        warpPerspective(mask, warp_mask, H, out_size);

        if (visual_dbg) {
            imshow("warp " + img.path, warp_img);
        }
        warp_imgs.push_back(warp_img);
        warp_masks.push_back(warp_mask);
    }
    overlap.setTo(1, (overlap == 0));
    for (int i = 0; i < warp_imgs.size(); ++i) {
        if (i < warp_imgs.size() - 1) {
            // caculate two image overlap
            cv::Mat overlay = (warp_masks[i] == warp_masks[i+1]);
            overlay.setTo(0, warp_masks[i] == 0);
            overlay.setTo(0, warp_masks[i+1] == 0);

            // imshow("overlay for " + std::to_string(i), overlay);
            // warp_mask[i] -= overlay;
            // caculate overlap corners
            int min_x = -1, max_x = -1;
            uchar *p;
            for (int y = 0; y < overlay.rows; ++y) {
                p = overlay.ptr<uchar >(y);
                for (int x = 0; x < overlay.cols; ++x) {
                    if (255 == p[x]) {
                        if (min_x == -1) {
                            min_x = x;
                            max_x = x;
                        }
                        if (x < min_x) {
                            min_x = x;
                        }
                        if (x > max_x) {
                            max_x = x;
                        }
                    }
                }
            }
            // std::cout << "overlay:" << overlay << std::endl;
            // std::cout << "min_x:" << min_x << " max_x:" << max_x << std::endl;
            int middle_x = (min_x + max_x) / 2;
            int w = warp_masks[i].cols;
            int h = warp_masks[i].rows;
            warp_masks[i](Rect(middle_x, 0, w-middle_x, h)).setTo(0);
            warp_masks[i+1](Rect(0, 0, middle_x, h)).setTo(0);
        }
        if (visual_dbg) {
            imshow("seam mask " + std::to_string(i), warp_masks[i]);
        }
    }

    cv::Mat stitch;
    if (blend_method == "mbb") {
        stitch = StitchWithMbbBlend(warp_imgs, warp_masks);
    } else if (blend_method == "no") {
        stitch = StitchWithNoBlend(warp_imgs, warp_masks);
    }
    imshow("stitch", stitch);
    const string img_name = "stitch_" + blend_method + ".jpg";
    imwrite(img_name, stitch);

    cv::waitKey(0);
    return 0;
}