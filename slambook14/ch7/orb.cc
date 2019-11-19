//
// Created by jerett on 2019/11/11.
//

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


int main(int argc, char *argv[]) {
    // std::cout << "fuck fuck fuck" << std::endl;
    auto img1 = cv::imread("1.png");
    auto img2 = cv::imread("2.png");

    std::vector<cv::KeyPoint> img1_keypoints;
    std::vector<cv::KeyPoint> img2_keypoints;
    cv::Mat img1_desc, img2_desc;

    auto orb_feature = cv::ORB::create();
    orb_feature->detect(img1, img1_keypoints);
    orb_feature->compute(img1, img1_keypoints, img1_desc);
    orb_feature->detect(img2, img2_keypoints);
    orb_feature->compute(img2, img2_keypoints, img2_desc);

    cv::Mat img1_draw, img2_draw;
    cv::drawKeypoints(img1, img1_keypoints, img1_draw);
    cv::drawKeypoints(img2, img2_keypoints, img2_draw);

    cv::imshow("img1", img1_draw);
    cv::imshow("img2", img2_draw);

    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher->match(img1_desc, img2_desc, matches);

    cv::Mat match_draw;
    cv::drawMatches(img1, img1_keypoints, img2, img2_keypoints, matches, match_draw);
    cv::imshow("draw match", match_draw);


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
    cv::Mat good_match_draw;
    cv::drawMatches(img1, img1_keypoints, img2, img2_keypoints, good_matches, good_match_draw);
    cv::imshow("draw good match", good_match_draw);
    cv::waitKey(0);

    return 0;
}
