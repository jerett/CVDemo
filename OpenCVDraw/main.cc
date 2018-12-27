//
// Created by jerett on 2018-12-26.
//


/**
 *
 * some opencv draw samples.
 *
 */


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

void TestDrawText() {
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_DEBUG);
    String text = "Funny text inside the box";
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 2;
    int thickness = 3;

    Mat img(600, 800, CV_8UC3, Scalar::all(0));

    int baseline = 0;
    Size textSize = getTextSize(text, fontFace,
                                fontScale, thickness, &baseline);
    CV_LOG_DEBUG(NULL, "baseline:" << baseline);
    baseline += thickness;

    // center the text
    Point text_org((img.cols - textSize.width) / 2,
                   (img.rows + textSize.height) / 2);
    circle(img, text_org, 5, Scalar(0, 0, 255), 2);

    // draw the box
    Point pt1 = text_org + Point(0, baseline);
    Point pt2 = text_org + Point(textSize.width, -textSize.height);
    circle(img, pt1, 5, Scalar(0, 255, 0), 2);
    circle(img, pt2, 5, Scalar(0, 255, 0), 2);
    rectangle(img, pt1, pt2, Scalar(0, 0, 255));
    // ... and the baseline first
    pt1 = text_org + Point(0, thickness);
    pt2 = text_org + Point(textSize.width, thickness);
    circle(img, pt1, 5, Scalar(255, 0, 0), 2);
    circle(img, pt2, 5, Scalar(255, 0, 0), 2);
    line(img, pt1, pt2, Scalar(0, 0, 255));

    putText(img, text, text_org, fontFace, fontScale,
            Scalar::all(255), thickness, 8);
    cv::imshow("Text Demo", img);
    cv::waitKey(0);
}

int main(int argc, char *argv[]) {
    TestDrawText();
    return 0;
}
