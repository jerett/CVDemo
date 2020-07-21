#include "ISP.h"
#include <opencv2/xphoto.hpp>
#include "RedundantDXTDenoise.h"


int RAW2JPG(const cv::Mat &rawImg, const DNGMetadata& metadata, cv::Mat& bgrImg) {
	//cv::Mat rawImg = file.GetImage();
	//cv::Mat yImg;
	//Raw28Bit(rawImg, yImg);

	cv::Mat rawOut;
	SubtractBlackLevel(metadata, rawImg, rawOut);

	double maxVal;
	cv::minMaxLoc(rawOut, 0, &maxVal, 0, 0);
	//std::cout << "maxVal:" << maxVal << std::endl;

	cv::Mat rawOut2;
	WhiteBalance(metadata, rawOut, rawOut2);

	cv::Mat bgr;
	Demosaic(metadata, rawOut2, bgr);

	//cv::Mat bgr2;
	//Denoise(bgr, bgr2);

	cv::Mat bgr3;
	ColorCorrect(metadata, bgr, bgr3);

	cv::Mat bgr4;
	GammaCorrectBGR24(metadata, bgr3, bgr4, 2.2);
	//bgrImg = bgr3;

	cv::Mat bgr5;
	Enhance(bgr4, bgr5);

	cv::Mat bgr6;
	Denoise(bgr5, bgr6);

	bgrImg = bgr6;
	//bgrImg = bgr5;
	return 0;
}

void SubtractBlackLevel(const DNGMetadata &file, const cv::Mat& rawIn, cv::Mat &rawOut) {
	const cv::Matx22f& blackLevels_ = file.GetBlackLevels();
	const int whiteLevel_ = file.GetWhiteLevel();
	// strecth pixel from [0, whilteLevel-blackLevel] to [0, whileteLevel]
	cv::Matx22f gain;
	for (int y = 0; y < 2; ++y) {
		for (int x = 0; x < 2; ++x) {
			gain(y, x) = static_cast<float>(whiteLevel_) / (whiteLevel_ - blackLevels_(y, x));
		}
	}

	rawOut.create(rawIn.size(), rawIn.type());

	const uint16_t* rawInData = rawIn.ptr<uint16_t>();
	uint16_t* rawOutData = rawOut.ptr<uint16_t>();

	const int w = rawIn.cols;
	const int h = rawIn.rows;
	for (int j = 0; j < h; j += 2) {
		for (int i = 0; i < w; i += 2) {
			// 2x2 block
			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 2; ++x) {
					int idx = (j + y) * w + i + x;
					int v = (rawInData[idx] - static_cast<int>(blackLevels_(y, x))) * gain(y, x);
					v = std::min(std::max(v, 0), whiteLevel_);
					rawOutData[idx] = static_cast<uint16_t>(v);
				}
			}
		}
	}
}

void WhiteBalance(const DNGMetadata &file, const cv::Mat& rawIn, cv::Mat& rawOut) {
	const cv::Vec3f WBGain_ = file.GetWBGain();
	const int whiteLevel_ = file.GetWhiteLevel();
	rawOut.create(rawIn.size(), rawIn.type());

	const uint16_t* rawInData = rawIn.ptr<uint16_t>();
	uint16_t* rawOutData = rawOut.ptr<uint16_t>();

	const int w = rawIn.cols;
	const int h = rawIn.rows;
	for (int j = 0; j < h; j += 2) {
		for (int i = 0; i < w; i += 2) {
			// 2x2 block
			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 2; ++x) {
					PixelType pixType = file.GetPixelType(y, x);
					int idx = (j + y) * w + i + x;;
					float g = WBGain_(int(pixType));
					float v = rawInData[idx] / g;
					v = std::min(static_cast<int>(v), whiteLevel_);
					//if (v > whiteLevel) {
					//	std::cout << "fuck fuck fuck" << std::endl;
					//}
					rawOutData[idx] = static_cast<uint16_t>(v);
				}
			}
		}
	}
}

void Demosaic(const DNGMetadata &file, const cv::Mat &rawIn, cv::Mat &bgrOut) {
	const BayerPattern bayerPattern_ = file.GetBayerPattern();
	const uint16_t whiteLevel_ = file.GetWhiteLevel();
	bgrOut.create(rawIn.size(), CV_16UC3);

	cv::ColorConversionCodes code;
	if (bayerPattern_ == GR_BG) {
		code = cv::COLOR_BayerGB2BGR;
	} else if (bayerPattern_ == RG_GB) {
		code = cv::COLOR_BayerBG2BGR;
	} else if (bayerPattern_ == BG_GR) {
		code = cv::COLOR_BayerRG2BGR;
	} else if (bayerPattern_ == GB_RG) {
		code = cv::COLOR_BayerGR2BGR;
	}
	
	cv::cvtColor(rawIn, bgrOut, code);

	uint16_t* bgrData = bgrOut.ptr<uint16_t>();
	const int w = rawIn.cols;
	const int h = rawIn.rows;
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			int idx = j  * w + i;
			bgrData[3 * idx] = std::min<uint16_t>(bgrData[3 * idx], whiteLevel_);
			bgrData[3 * idx + 1] = std::min<uint16_t>(bgrData[3 * idx + 1], whiteLevel_);
			bgrData[3 * idx + 2] = std::min<uint16_t>(bgrData[3 * idx + 2], whiteLevel_);
		}
	}
}


void ColorCorrect(const DNGMetadata& file, const cv::Mat& bgrIn, cv::Mat& bgrOut) {
	bgrOut.create(bgrIn.size(), CV_16UC3);
	const int whiteLevel_ = file.GetWhiteLevel();

	const cv::Matx33f cameraFromXYZ = file.GetCameraFromXYZColorMatrix();
	const cv::Matx33f xyzFromSRGB = file.GetXYZFromSRGBColorMatrix();
	cv::Matx33f cameraFromSRGB = cameraFromXYZ * xyzFromSRGB;
	// norm 
	cv::Mat sumRowVec;
	for (int i = 0; i < 3; ++i) {
		float sum = .0f;
		for (int j = 0; j < 3; ++j) {
			sum += cameraFromSRGB(i, j);
		}
		//std::cout << "row[" << i << "]" << " sum:" << sum << std::endl;
		for (int j = 0; j < 3; ++j) {
			cameraFromSRGB(i, j) /= sum;
		}
	}
	cv::Matx33f srgb2camera = cameraFromSRGB.inv();
	//std::cout << "srgb2camera matrix:\n" << srgb2camera << std::endl;

	const uint16_t* bgrInData = bgrIn.ptr<uint16_t>();
	uint16_t* bgrOutData = bgrOut.ptr<uint16_t>();

	const int w = bgrIn.cols;
	const int h = bgrIn.rows;
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			int idx = j  * w + i;
			const cv::Vec3f rgb(bgrInData[3 * idx + 2], bgrInData[3 * idx + 1], bgrInData[3 * idx]);
			cv::Vec3i rgbOut = static_cast<cv::Vec3i>(srgb2camera * static_cast<cv::Vec3f>(rgb));

			bgrOutData[3 * idx] = static_cast<uint16_t>(std::max(std::min(rgbOut[2], whiteLevel_), 0));
			bgrOutData[3 * idx + 1] = static_cast<uint16_t>(std::max(std::min(rgbOut[1], whiteLevel_), 0));
			bgrOutData[3 * idx + 2] = static_cast<uint16_t>(std::max(std::min(rgbOut[0], whiteLevel_), 0));
			//bgrInData[3 * idx] = std::min<uint16_t>(bgrData[3 * idx], whiteLevel);
			//bgrData[3 * idx + 1] = std::min<uint16_t>(bgrData[3 * idx + 1], whiteLevel);
			//bgrData[3 * idx + 2] = std::min<uint16_t>(bgrData[3 * idx + 2], whiteLevel);
		}
	}
}


void GammaCorrectBGR24(const DNGMetadata& file, const cv::Mat& bgrIn, cv::Mat &bgrOut, float gamma) {
	const int whiteLevel_ = file.GetWhiteLevel();
	const int maxVal = 255;

	bgrOut.create(bgrIn.size(), CV_8UC3);

	float invGamma = 1.0 / gamma;
	std::vector<uint8_t> lut(whiteLevel_ + 1);
	for (int i = 0; i < whiteLevel_ + 1; ++i) {
		float tmp = static_cast<float>(i) / whiteLevel_;
		tmp = std::powf(tmp, invGamma) * maxVal;
		if (tmp > maxVal) {
			tmp = maxVal;
		}
		if (tmp < 0) {
			tmp = 0;
		}
		lut[i] = static_cast<uint8_t>(tmp + 0.5f);
		//std::cout << "lut[" << i << "]" << " " << lut[i] << std::endl;
	}
	//std::cout << "lut sz:" << lut.size() << std::endl;

	const uint16_t* bgrInData = bgrIn.ptr<uint16_t>();
	uint8_t* bgrOutData = bgrOut.ptr<uint8_t>();
	const int w = bgrIn.cols;
	const int h = bgrIn.rows;
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			int idx = j  * w + i;
			uint16_t b = bgrInData[3 * idx];
			uint16_t g = bgrInData[3 * idx + 1];
			uint16_t r = bgrInData[3 * idx + 2];
			//std::cout << "b:" << b << " g:" << g << " r:" << r << std::endl;
			bgrOutData[3 * idx] = lut[b];
			bgrOutData[3 * idx + 1] = lut[g];
			bgrOutData[3 * idx + 2] = lut[r];
		}
	}
}


int brightnessConstrast(const cv::Mat& src, cv::Mat& dst, float brightness, float percent) {
	int width = src.cols;
	int height = src.rows;
	// Scalar meanVal = cv::mean(src);
	if (src.type() == CV_8UC3) {
		cv::Scalar meanVal(127.5f, 127.5f, 127.5f);
		for (int y = 0; y < height; y++) {
			cv::Vec3b* dstPtr = dst.ptr<cv::Vec3b>(y);
			const cv::Vec3b* srcPtr = src.ptr<cv::Vec3b>(y);
			for (int x = 0; x < width; x++) {
				cv::Vec3b& dstVal = dstPtr[x];
				const cv::Vec3b& srcVal = srcPtr[x];
				dstVal[0] = cv::saturate_cast<uchar>(meanVal[0] + ((int)srcVal[0] - meanVal[0]) * percent + brightness);
				dstVal[1] = cv::saturate_cast<uchar>(meanVal[1] + ((int)srcVal[1] - meanVal[1]) * percent + brightness);
				dstVal[2] = cv::saturate_cast<uchar>(meanVal[2] + ((int)srcVal[2] - meanVal[2]) * percent + brightness);
			}
		}
	} else if (src.type() == CV_16UC3) {
		cv::Scalar meanVal(511.5f, 511.5f, 511.5f);
		for (int y = 0; y < height; y++) {
			cv::Vec3w* dstPtr = dst.ptr<cv::Vec3w>(y);
			const cv::Vec3w* srcPtr = src.ptr<cv::Vec3w>(y);
			for (int x = 0; x < width; x++) {
				cv::Vec3w& dstVal = dstPtr[x];
				const cv::Vec3w& srcVal = srcPtr[x];
				dstVal[0] = cv::saturate_cast<ushort>(meanVal[0] + ((int)srcVal[0] - meanVal[0]) * percent + brightness);
				dstVal[1] = cv::saturate_cast<ushort>(meanVal[1] + ((int)srcVal[1] - meanVal[1]) * percent + brightness);
				dstVal[2] = cv::saturate_cast<ushort>(meanVal[2] + ((int)srcVal[2] - meanVal[2]) * percent + brightness);
			}
		}
	}
	return 0;
}


void Split4(const cv::Mat& rawIn, std::vector<cv::Mat>& imgsOut) {
	//const int w = rawIn.cols;
	//const int h = rawIn.rows;

	//imgsOut.resize(4);
	//for (int i = 0; i < 4; ++i) {
	//	imgsOut.
	//}


	//for (int j = 0; j < h; ++j) {
	//	for (int i = 0; i < w; ++i) {

	//	}
	//}

}

void SplitImage2Vec(const cv::Mat& img, int numW, int numH, std::vector<cv::Mat>& vImgs) {
	// CV_Assert(img.type()==CV_8U);
	vImgs.resize(numW * numH);
	int width = img.cols / numW;
	int height = img.rows / numH;
	std::vector<uchar*> pImages;
	vImgs.resize(numW * numH);
	for (auto& vImg : vImgs) {
		vImg.create(height, width, img.type());
		pImages.push_back(vImg.data);
	}

	uchar* imgData = (uchar*)img.data;
	int cn = img.channels();
	//auto fun = [cn, numH, numW, &img, width, imgData, &pImages](int sy, int ey) {
	for (int i1 = 0; i1 < height; i1++) {
		for (int j1 = 0; j1 < width; ++j1) {
			int id1 = i1 * width + j1;
			for (int i = 0; i < numH; ++i) {
				int idy = (i + i1 * numH) * img.cols;
				for (int j = 0; j < numW; ++j) {
					int id = i * numW + j;
					int id2 = idy + j + j1 * numW;
					for (int k = 0; k < cn; ++k) {
						pImages[id][id1 * cn + k] = imgData[id2 * cn + k];
					}
				}
			}
		}
	}
}

void MergeVecImage(cv::Mat& img, int numW, int numH, const std::vector<cv::Mat>& vImgs) {
	// CV_Assert(vImgs[0].type() == CV_8U);
	int width = vImgs[0].cols;
	int height = vImgs[0].rows;
	//img.create(cv::Size(numW * width, numH * height), vImgs[0].type());
	uchar* imgData = (uchar*)img.data;
	int cn = img.channels();
	for (int i1 = 0; i1 < height; i1++) {
		for (int j1 = 0; j1 < width; ++j1) {
			int id1 = i1 * width + j1;
			for (int i = 0; i < numH; ++i) {
				int idy = (i + i1 * numH) * img.cols;
				for (int j = 0; j < numW; ++j) {
					int id = i * numW + j;
					int id2 = idy + j + j1 * numW;
					for (int k = 0; k < cn; ++k) {
						imgData[id2 * cn + k] = vImgs[id].data[id1 * cn + k];
					}
				}
			}
		}
	}
}


void Denoise(const cv::Mat& bgrIn, cv::Mat& bgrOut) {
	cv::Mat yuv;
	cv::cvtColor(bgrIn, yuv, cv::COLOR_BGR2YUV);

	std::vector<cv::Mat> yuvChannels(3);
	cv::split(yuv, yuvChannels);

	RedundantDXTDenoise denoise;
	denoise(yuvChannels[0], yuvChannels[0], 10);
	const int numWSplit = 4;
	const int numHSplit = 4;
	std::vector<cv::Mat> vImgs;

	{
		SplitImage2Vec(yuvChannels[1], numWSplit, numHSplit, vImgs);
		for (int j = 0; j < numHSplit; ++j) {
			for (int i = 0; i < numWSplit; ++i) {
				int idx = j * numWSplit + i;
				denoise(vImgs[idx], vImgs[idx], 20);
				//cv::bilateralFilter(vImgs[idx], vImgs[idx], 9, 75, 75, cv::BORDER_DEFAULT);
			}
		}
		MergeVecImage(yuvChannels[1], numWSplit, numHSplit, vImgs);
	}

	{
		SplitImage2Vec(yuvChannels[2], numWSplit, numHSplit, vImgs);
		for (int j = 0; j < numHSplit; ++j) {
			for (int i = 0; i < numWSplit; ++i) {
				int idx = j * numWSplit + i;
				denoise(vImgs[idx], vImgs[idx], 20);
			}
		}
		MergeVecImage(yuvChannels[2], numWSplit, numHSplit, vImgs);
		cv::merge(yuvChannels, yuv);
	}
	cv::cvtColor(yuv, bgrOut, cv::COLOR_YUV2BGR);
}

void Enhance(const cv::Mat& bgrIn, cv::Mat& bgrOut) {
	// sharpen image using "unsharp mask" algorithm
	cv::Mat blurred; double sigma = 3, threshold = 5, amount = 1.3;
	GaussianBlur(bgrIn, blurred, cv::Size(), sigma, sigma);
	cv::Mat lowContrastMask = abs(bgrIn - blurred) < threshold;
	bgrOut = bgrIn * (1 + amount) + blurred * (-amount);
	bgrIn.copyTo(bgrOut, lowContrastMask); 

	// hsv
	cv::Mat hsv;
	cv::cvtColor(bgrOut, hsv, cv::COLOR_BGR2HSV);
	const unsigned char hShift = 0;
	//const unsigned char sShift = 30;
	//const unsigned char vShift = 10;
	const unsigned char sShift = 30;
	const unsigned char vShift = 0;

	uint8_t* hsvData = hsv.ptr<uint8_t>();
	for (int j = 0; j < hsv.rows; j++) {
		for (int i = 0; i < hsv.cols; i++) {
			int idx = j * hsv.cols + i;
			//uint8_t h = hsv
			uint8_t h = hsvData[3 * idx];
			uint8_t s = hsvData[3 * idx + 1];
			uint8_t v = hsvData[3 * idx + 2];

			uint16_t sNew = s + sShift;
			uint16_t vNew = v + vShift;
			if (sNew >= 255) sNew = 255;
			if (vNew >= 255) vNew = 255;
			hsvData[3 * idx + 1] = static_cast<uint8_t>(sNew);
			hsvData[3 * idx + 2] = static_cast<uint8_t>(vNew);
		}
	}
	cvtColor(hsv, bgrOut, cv::COLOR_HSV2BGR);

	// brightness, contrast
	brightnessConstrast(bgrOut, bgrOut, 30, 1.2);
}

