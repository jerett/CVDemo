#include "MosseTracker.h"



MOSSETracker::MOSSETracker() {
}

void ComplexDivide(const cv::Mat &A, const cv::Mat &B, cv::Mat &C) {
	//cv::Mat a1, a2, b1, b2, c1, c2;
	cv::Mat a[2], b[2], c[2];
	cv::split(A, a);
	cv::split(B, b);
	
	cv::Mat denom = b[0].mul(b[0]) + b[1].mul(b[1]);
	cv::Mat real = a[0].mul(b[0]) + a[1].mul(b[1]);
	cv::Mat im = a[1].mul(b[0]) - a[0].mul(b[1]);
	cv::divide(real, denom, real);
	cv::divide(im, denom, im);

	cv::Mat chn[] = { real, im };
	cv::merge(chn, 2, C);
}

cv::Mat divDFTs(const cv::Mat& src1, const cv::Mat& src2) {
	cv::Mat c1[2], c2[2], a1, a2, s1, s2, denom, re, im;

	// split into re and im per src
	cv::split(src1, c1);
	cv::split(src2, c2);

	// (Re2*Re2 + Im2*Im2) = denom
	//   denom is same for both channels
	cv::multiply(c2[0], c2[0], s1);
	cv::multiply(c2[1], c2[1], s2);
	cv::add(s1, s2, denom);

	// (Re1*Re2 + Im1*Im1)/(Re2*Re2 + Im2*Im2) = Re
	cv::multiply(c1[0], c2[0], a1);
	cv::multiply(c1[1], c2[1], a2);
	cv::divide(a1 + a2, denom, re, 1.0);

	// (Im1*Re2 - Re1*Im2)/(Re2*Re2 + Im2*Im2) = Im
	cv::multiply(c1[1], c2[0], a1);
	cv::multiply(c1[0], c2[1], a2);
	cv::divide(a1 + a2, denom, im, -1.0);

	// Merge Re and Im back into a complex matrix
	cv::Mat dst, chn[] = { re,im };
	cv::merge(chn, 2, dst);
	return dst;
}


cv::Mat RandomAffine(const cv::Mat& img, cv::RNG &rng) {
	cv::Mat affineImg;
	double angle = rng.uniform(-180.0 / 16, 180.0 / 16);
	//double angle = 10;
	//double angle = 0;
	cv::Point2f center(img.cols / 2, img.rows / 2);
	double scale = rng.uniform(0.9, 1.1);
	//double scale = 0.9;
	cv::Mat affine = cv::getRotationMatrix2D(center, angle, scale);
	std::cout << "angle:" << angle << " scale:" << scale << std::endl;
	cv::warpAffine(img, affineImg, affine, img.size());
	//cv::imshow("affine", affineImg);
	//cv::waitKey(0);
	return affineImg;
}

cv::Mat randWarp(const cv::Mat& a) {
	cv::RNG rng(8031965);

	// random rotation
	double C = 0.1;
	double ang = rng.uniform(-C, C);
	double c = cos(ang), s = sin(ang);
	// affine warp matrix
	cv::Mat_<float> W(2, 3);
	W << c + rng.uniform(-C, C), -s + rng.uniform(-C, C), 0,
		s + rng.uniform(-C, C), c + rng.uniform(-C, C), 0;

	// random translation
	cv::Mat_<float> center_warp(2, 1);
	center_warp << a.cols / 2, a.rows / 2;
	W.col(2) = center_warp - (W.colRange(0, 2)) * center_warp;

	cv::Mat warped;
	warpAffine(a, warped, W, a.size(), cv::BORDER_REFLECT);
	return warped;
}

cv::Mat Preprocess(const cv::Mat& inMat) {
	cv::Mat outMat;
	inMat.convertTo(outMat, CV_32F);
	// log
	cv::log(outMat + 1, outMat);
	// norm
	cv::Scalar mean, std;
	cv::meanStdDev(outMat, mean, std);
	const double eps = 1e-6;
	outMat = (outMat - mean[0]) / (std[0] + eps);
	// cos window
	cv::Mat hann;
	cv::createHanningWindow(hann, outMat.size(), CV_32F);
	outMat = outMat.mul(hann);
	return outMat;
}

bool MOSSETracker::Init(const cv::Mat& img, const cv::Rect2d& box) {
	cv::Mat grayImg;
	if (img.channels() == 3) {
		cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	} else {
		grayImg = img;
	}
	const int w = box.width;
	const int h = box.height;

	//const int optimalW = cv::getOptimalDFTSize(w);
	//const int optimalH = cv::getOptimalDFTSize(h);

	center_.x = box.x + box.width / 2;
	center_.y = box.y + box.height / 2;
	size_.width = w;
	size_.height = h;
	//cv::Mat window = grayImg(box);
	cv::Mat window;
	cv::getRectSubPix(grayImg, size_, center_, window);
	//cv::imwrite("initbox.jpg", window);

	// init response g
	cv::Mat g = cv::Mat::zeros(size_, CV_32F);

	g.at<float>(h/2, w/2) = 1.0f;
	cv::GaussianBlur(g, g, cv::Size(-1, -1), 2.0f);

	cv::Point maxLoc;
	double maxVal;
	cv::minMaxLoc(g, nullptr, &maxVal, nullptr, &maxLoc);
	//std::cout << "maxVal:" << maxVal << std::endl;
	g /= maxVal;
	cv::imshow("g", g);
	cv::dft(g, G_, cv::DFT_COMPLEX_OUTPUT);

	// init train
	A_ = cv::Mat::zeros(G_.size(), G_.type());
	B_ = cv::Mat::zeros(G_.size(), G_.type());
	// rand warp
	int N = 8;
	cv::RNG rng(0);
	cv::Mat f = Preprocess(window);
	for (int i = 0; i < N; ++i) {
		cv::Mat fi = RandomAffine(f, rng);
		//cv::Mat transformImg = randWarp(window);
		//cv::Mat fi = Preprocess(transformImg);
		cv::Mat Ai, Bi;
		cv::Mat Fi;
		cv::dft(fi, Fi, cv::DFT_COMPLEX_OUTPUT);
		cv::mulSpectrums(G_, Fi, Ai, 0, true);
		cv::mulSpectrums(Fi, Fi, Bi, 0, true);
		A_ += Ai;
		B_ += Bi;
	}
	return true;
}

bool MOSSETracker::Update(const cv::Mat& img, cv::Rect2d& outbox) {
	cv::Mat grayImg;
	if (img.channels() == 3) {
		cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	} else {
		grayImg = img;
	}

	float psr = Detect(grayImg, outbox);
	Train(grayImg, outbox);
	++frameCnt;
	cv::waitKey(0);
	return true;
}

float MOSSETracker::Detect(const cv::Mat& img, cv::Rect2d& outbox) {
	cv::Mat window;
	cv::getRectSubPix(img, size_, center_, window);
	//std::string filename = "frame" + std::to_string(frameCnt) + "_window0.jpg";
	//cv::imwrite(filename, window);
	//cv::imshow("window0", window);
	window = Preprocess(window);
	cv::Mat F;
	cv::dft(window, F, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat RESPONSE;
	ComplexDivide(A_, B_, Hconj_);
	cv::mulSpectrums(F, Hconj_, RESPONSE, 0, false);
	//Hconj_ = divDFTs(A_, B_);
	//cv::mulSpectrums(F, Hconj_, RESPONSE, 0, true);
	cv::Mat response;
	cv::idft(RESPONSE, response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

	cv::Point maxLoc;
	double maxVal;
	double minVal;
	cv::minMaxLoc(response, &minVal, &maxVal, nullptr, &maxLoc);
	response /= maxVal;
	cv::imshow("g", response);

	// caculate psr
	cv::Scalar mean, std;
	cv::meanStdDev(response, mean, std);
	double psr = (maxVal - minVal) / (std[0] + 1e-6);
	//if (psr < 5.8) return false;
	std::cout << "psr:" << psr << std::endl;

	// update pos
	float dx = (maxLoc.x - response.size().width / 2);
	float dy = (maxLoc.y - response.size().height / 2);
	center_.x += dx;
	center_.y += dy;
	outbox.x = center_.x - size_.width / 2;
	outbox.y = center_.y - size_.height / 2;
	outbox.width = size_.width;
	outbox.height = size_.height;
	//std::cout << "framecnt:" << frameCnt <<  " maxLoc:" << maxLoc << " dx:" << dx << " dy:" << dy << std::endl;
	//std::cout << "center:" << center_ << std::endl;
	return psr;
}

bool MOSSETracker::Train(const cv::Mat& img, cv::Rect2d& outbox) {
	cv::Mat newWindow;
	cv::Mat F;
	cv::Mat newA, newB;
	cv::getRectSubPix(img, size_, center_, newWindow);
	//std::string filename = "frame" + std::to_string(frameCnt) + "_window1.jpg";
	//cv::imwrite(filename, newWindow);
	//cv::imshow("window1", newWindow);
	//cv::imwrite("initbox.jpg", window);
	cv::Mat f = Preprocess(newWindow);
	cv::dft(f, F, cv::DFT_COMPLEX_OUTPUT);
	cv::mulSpectrums(G_, F, newA, 0, true);
	cv::mulSpectrums(F, F, newB, 0, true);
	A_ = rate_ * newA + (1 - rate_) * A_;
	B_ = rate_ * newB + (1 - rate_) * B_;
	return true;
}



