
#pragma once

#include "DNGFile.h"
#include <opencv2/opencv.hpp>


int RAW2JPG(const cv::Mat &rawImg, const DNGMetadata& metadata, cv::Mat& bgrImg);

void SubtractBlackLevel(const DNGMetadata& metadata, const cv::Mat& rawIn, cv::Mat& rawOut);
void WhiteBalance(const DNGMetadata& metadata, const cv::Mat& rawIn, cv::Mat& rawOut);
void Demosaic(const DNGMetadata& metadata, const cv::Mat& rawIn, cv::Mat& bgrOut);
void ColorCorrect(const DNGMetadata& metadata, const cv::Mat& bgrIn, cv::Mat& bgrOut);
void GammaCorrectBGR24(const DNGMetadata& metadata, const cv::Mat& bgrIn, cv::Mat& bgrOut, float gamma);
void Denoise(const cv::Mat& bgrIn, cv::Mat& bgrOut);
void Enhance(const cv::Mat& bgrIn, cv::Mat& bgrOut);

