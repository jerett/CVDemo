
#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "DNGFile.h"

//int HDRPlus(const std::vector<std::shared_ptr<DNGFile>>& dngFiles, cv::Mat &bgrImg);
int HDRPlus(const std::vector<cv::Mat>& rawImgs, 
			const std::vector<std::shared_ptr<DNGMetadata>>& imgsMetadata,
			cv::Mat &bgrImg);

