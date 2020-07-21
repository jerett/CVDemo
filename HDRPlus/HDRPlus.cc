
#include "HDRPlus.h"
#include "ISP.h"

void Raw28Bit(const cv::Mat& rawImg, cv::Mat& yImg) {
	int w = rawImg.cols;
	int h = rawImg.rows;

	//yImg.create(h / 2, w / 2, CV_16UC1);
	yImg.create(h / 2, w / 2, CV_8UC1);

	const uint16_t* rawImgData = rawImg.ptr<uint16_t>();
	uint8_t* yImgData = yImg.ptr<uint8_t>();

	for (int j = 0; j < h; j += 2) {
		for (int i = 0; i < w; i += 2) {
			// 2x2 block
			uint16_t y16 = 0;
			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 2; ++x) {
					int idx = (j + y) * w + i + x;
					int idx2 = j / 2 * w / 2 + i/2;
					y16 += (rawImgData[idx] >> 2);
					yImgData[idx2] = (y16 >> 2);
					//int v = (rawInData[idx] - static_cast<int>(blackLevels(y, x))) * gain(y, x);
					//v = std::min(std::max(v, 0), whiteLevel);
					//rawOutData[idx] = static_cast<uint16_t>(v);
				}
			}
		}
	}
}

cv::Mat DrawFlow(const cv::Mat& flow) {
	cv::Mat flow_uv[2];
	cv::Mat mag, ang;
	cv::Mat hsv_split[3], hsv, rgb;

	split(flow, flow_uv);
	cv::multiply(flow_uv[1], -1, flow_uv[1]);
	cv::cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
	cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
	hsv_split[0] = ang;
	hsv_split[1] = mag;
	hsv_split[2] = cv::Mat::ones(ang.size(), ang.type());
	merge(hsv_split, 3, hsv);
	// std::cout << "hsv type:" << hsv.type() << std::endl;
	cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
	return rgb;
}

void CreateAlignMap(const cv::Mat& flow, cv::Mat &alignMap) {
	alignMap.create(flow.size() * 2, CV_32FC2);
	//weightsMap.create(flow.size() * 2, CV_32FC1);
	//weightsMap.setTo(1.0 / N);

	int w = alignMap.cols;
	int h = alignMap.rows;

	const float* flowData = flow.ptr<float>();
	float* mapData = alignMap.ptr<float>();

	for (int j = 0; j < h; j += 2) {
		for (int i = 0; i < w; i += 2) {
			int idx = j / 2 * w / 2 + i/2;
			float dx = flowData[2 * idx];
			float dy = flowData[2 * idx + 1];

			int mapX = round(dx + i/2) * 2;
			int mapY = round(dy + j/2) * 2;
			//std::cout << "dx:" << mapX - i << "  dy:" << mapY - j << std::endl;

			//std::cout << "dx dy" << dx << "  " << dy << std::endl;
			// 2x2 block
			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 2; ++x) {
					int idx = (j + y) * w + i + x;
					mapData[2 * idx] = mapX + x;
					mapData[2 * idx + 1] = mapY + y;
					//mapData[2 * idx] = (i + x);
					//mapData[2 * idx + 1] = (j + y);
				}
			}
		}
	}
}

void RobustMerge(int refIdx,
				 const std::vector<cv::Mat>& remapRawImgs,
				 cv::Mat& rawOut) {
	const cv::Mat& refRawImg = remapRawImgs[refIdx];
	rawOut = cv::Mat::zeros(refRawImg.size(), refRawImg.type());
	cv::Mat mergeTmp(refRawImg.size(), CV_32FC1);
	cv::Mat totalWeight = cv::Mat::zeros(refRawImg.size(), CV_32FC1);
	const int N = remapRawImgs.size();

	int w = refRawImg.cols;
	int h = refRawImg.rows;

	const uint16_t* refRawData = refRawImg.ptr<uint16_t>();
	uint16_t* rawOutData = rawOut.ptr<uint16_t>();
	float* mergeData = mergeTmp.ptr<float>();
	float* weightData = totalWeight.ptr<float>();

	for (int n = 0; n < N; ++n) {
		const cv::Mat& remapRaw = remapRawImgs[n];
		const uint16_t* rawData = remapRaw.ptr<uint16_t>();

		for (int j = 0; j < h; j += 2) {
			for (int i = 0; i < w; i += 2) {

				// 2x2 block
				int rawY = 0;
				int refY = 0;
				for (int y = 0; y < 2; ++y) {
					for (int x = 0; x < 2; ++x) {
						int idx = (j + y) * w + i + x;
						rawY += rawData[idx];
						refY += refRawData[idx];
					}
				}
				rawY = rawY >> 2;
				refY = refY >> 2;
				int diff = abs(rawY - refY);
				float weight = 1.0f;
				if (diff >= 0 && diff <= 20) {
					weight = 80;
					//weight = 1;
				} else if (diff <= 80) {
					weight = 80 - diff;
					//weight = 2;
					//std::cout << "2 diff:" << diff << std::endl;
				} else {
					weight = 0;
				}

				for (int y = 0; y < 2; ++y) {
					for (int x = 0; x < 2; ++x) {
						int idx = (j + y) * w + i + x;
						weightData[idx] += weight;
						//rawOutData[idx] += rawData[idx] * weight;
						mergeData[idx] += rawData[idx] * weight;
					}
				}
			}
		}
	}
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			int idx = j * w + i;
			rawOutData[idx] = mergeData[idx] / weightData[idx];
		}
	}
}


void AverageMerge(const std::vector<cv::Mat> &remapRawImgs, cv::Mat &rawOut) {
	rawOut = cv::Mat::zeros(remapRawImgs[0].size(),remapRawImgs[0].type());
	const int N = remapRawImgs.size();
	 //average all
	for (const cv::Mat& remapRaw : remapRawImgs) {
		rawOut += remapRaw;
	}
	rawOut /= N;
}


int StackFrame(const std::vector<cv::Mat>& rawImgs,
			   const std::vector<std::shared_ptr<DNGMetadata>>& imgsMetadata,
			   cv::Mat& rawOut) {
	const int N = rawImgs.size();

	std::vector<cv::Mat> yImgs;
	// align
	// raw->y, avg rggb -> y
	for (const cv::Mat& rawImg : rawImgs) {
		cv::Mat yImg;
		Raw28Bit(rawImg, yImg);
		yImgs.push_back(yImg);
	}
	// optical flow
	auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
	int refIndex = 0;
	cv::Mat refYImg = yImgs[refIndex];
	cv::Mat refRawImg = rawImgs[refIndex];
	
	// align
	const int width = refYImg.cols;
	const int height = refYImg.rows;
	std::vector<cv::Mat> remapRawImgs;
	std::vector<cv::Mat> weightMaps;

	for (int i = 0; i < yImgs.size(); ++i) {
		cv::Mat rawImg = rawImgs[i];

		if (i != refIndex) {
			cv::Mat flow;
			//dis->calc(yImgs[i], refImg, flow);
			dis->calc(refYImg, yImgs[i], flow);
			//extend flow to map
			cv::Mat alignMap;
			cv::Mat weightMap;
			//cv::Mat map = CreateAlignMap(flow);
			CreateAlignMap(flow, alignMap);
			
			cv::Mat remapRawImg;
			cv::remap(rawImg, remapRawImg, alignMap, cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REFLECT101);
			remapRawImgs.push_back(remapRawImg);
			//flowDrawMap.push_back(DrawFlow(flow));
			//flowMap.push_back(flow);
			//cv::remap()
		} else {
			remapRawImgs.push_back(rawImg);
		}
	}

	cv::Mat averageRawOut;
	AverageMerge(remapRawImgs, averageRawOut);
	RobustMerge(refIndex, remapRawImgs, rawOut);

	// for visual debug
	for (int i = 0; i < remapRawImgs.size(); ++i) {
		cv::Mat bgrImg;
		RAW2JPG(rawImgs[i], *imgsMetadata[i], bgrImg);
		cv::imwrite("img" + std::to_string(i) + ".jpg", bgrImg);

		RAW2JPG(remapRawImgs[i], *imgsMetadata[i], bgrImg);
		cv::imwrite("remapImg" + std::to_string(i) + ".jpg", bgrImg);
	}
	cv::Mat bgrImg;
	RAW2JPG(rawOut, *imgsMetadata[refIndex], bgrImg);
	cv::imwrite("outRobustMerge.jpg", bgrImg);

	cv::Mat bgrImg2;
	RAW2JPG(averageRawOut, *imgsMetadata[refIndex], bgrImg2);
	cv::imwrite("outAverageMerge.jpg", bgrImg2);
	return 0;
}

int HDRPlus(const std::vector<cv::Mat>& rawImgs,
			const std::vector<std::shared_ptr<DNGMetadata>>& imgsMetadata,
			cv::Mat& bgrImg) {
	cv::Mat rawOut;
	StackFrame(rawImgs, imgsMetadata, rawOut);
	return 0;
}

