//
// Created by jerett on 2019-03-18.
//

#include "util.h"
#include <algorithm>

namespace cd {

cv::Mat DetectPreprocess(const cv::Mat &in_img,
                         const cv::Size &net_size,
                         float *scale, float *pad_w, float *pad_h) {
    if (in_img.cols == net_size.width && in_img.rows == net_size.height) {
        return in_img;
    }
    if (static_cast<float>(in_img.cols) / net_size.width >= static_cast<float>(in_img.rows) / net_size.height) {
        *scale = static_cast<float>(net_size.width) / in_img.cols;
    } else {
        *scale = static_cast<float>(net_size.height) / in_img.rows;
    }
    cv::Mat resize_in_img;
    if (*scale != 1) {
        cv::resize(in_img, resize_in_img, cv::Size(), *scale, *scale);
    }
    *pad_w = (net_size.width - resize_in_img.cols) / 2.0f;
    *pad_h = (net_size.height - resize_in_img.rows) / 2.0f;
    cv::Mat dst;
    cv::copyMakeBorder(resize_in_img,
                       dst,
                       int(*pad_h),
                       int(*pad_h + 0.5),
                       int(*pad_w),
                       int(*pad_w + 0.5),
                       cv::BORDER_CONSTANT,
                       cv::Scalar(127, 127, 127));
    return dst;
}

cv::Rect2d RemapBoxOnSrc(const cv::Rect2d &box, const cv::Size &net_size, const cv::Size &img_size,
                         float scale, float pad_w, float pad_h) {
    float xmin = box.x;
    float ymin = box.y;
    float xmax = xmin + box.width;
    float ymax = ymin + box.height;
    cv::Rect2d remap_box;
    remap_box.x = std::max(.0f, (xmin - pad_w) / scale);
    remap_box.width = std::min(img_size.width - 1.0f, (xmax - pad_w) / scale) - remap_box.x;
    remap_box.y = std::max(.0f, (ymin - pad_h) / scale);
    remap_box.height = std::min(img_size.height - 1.0f, (ymax - pad_h) / scale) - remap_box.y;
    return remap_box;
}

}