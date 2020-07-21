#pragma once
#include <iostream>
#include <deque>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)


double YPSNR(cv::InputArray src1, cv::InputArray src2);
void addNoise(cv::InputArray src, cv::OutputArray dest, double sigma, double solt_papper_ratio = 0.0);

void cvtColorBGR2DCT3PLANE_32f(const cv::Mat& src, cv::Mat& dest);
void cvtColorPLANEDCT32BGR_32f(const cv::Mat& src, cv::Mat& dest);
enum
{
	TIME_AUTO = 0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};
class CalcTime
{
	int64 pre;
	std::string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode();
	std::vector<std::string> lap_mes;
public:

	void start();
	void setMode(int mode);
	void setMessage(std::string src);
	void restart();
	double getTime();
	void show();
	void show(std::string message);
	void lap(std::string message);
	void init(std::string message, int mode, bool isShow);

	CalcTime(std::string message, int mode = TIME_AUTO, bool isShow = true);
	CalcTime();

	~CalcTime();
};

void cvtColorBGR2PLANE(const cv::Mat& src, cv::Mat& dest);
void cvtColorPLANE2BGR(const cv::Mat& src, cv::Mat& dest);


class RedundantDXTDenoise
{
public:
	enum BASIS
	{
		DCT = 0,
		DHT = 1,
		DWT = 2//under construction
	};
	bool isSSE;

	void init(cv::Size size_, int color_, cv::Size patch_size_);
	RedundantDXTDenoise(cv::Size size, int color, cv::Size patch_size_ = cv::Size(8, 8));
	RedundantDXTDenoise();
	virtual void operator()(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), BASIS transform_basis = BASIS::DCT);

protected:
	float getThreshold(float sigmaNoise);
	BASIS basis;
	cv::Size patch_size;
	cv::Size size;
	cv::Mat buff;
	cv::Mat sum;

	cv::Mat im;

	int channel;

	virtual void body(float *src, float* dest, float Th);

	void div(float* inplace0, float* inplace1, float* inplace2, float* w0, float* w1, float* w2, const int size1);
	void div(float* inplace0, float* inplace1, float* inplace2, const int patch_area, const int size1);

	void div(float* inplace0, float* w0, const int size1);
	void div(float* inplace0, const int patch_area, const int size1);
};

class RRDXTDenoise : public RedundantDXTDenoise
{
public:
	enum SAMPLING
	{
		FULL = 0,
		LATTICE,
		POISSONDISK,
	};

	RRDXTDenoise();
	RRDXTDenoise(cv::Size size, int color, cv::Size patch_size_ = cv::Size(8, 8));
	

	void generateSamplingMaps(cv::Size imageSize, cv::Size patch_size, int number_of_LUT, int d, SAMPLING sampleType = SAMPLING::POISSONDISK);

	virtual void operator()(cv::Mat& src_, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), BASIS transform_basis = BASIS::DCT);
	void colorredundunt(cv::Mat& src_, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), BASIS transform_basis = BASIS::DCT);
	cv::RNG rng;
protected:
	void div(float* inplace0, float* inplace1, float* inplace2, float* count, const int size1);
	void div(float* inplace0, float* inplace1, float* inplace2, float* inplace3, float* count, const int size1);

	virtual void body(float *src, float* dest, float Th);


	void getSamplingFromLUT(cv::Mat& samplingMap);
	void setSamplingMap(cv::Mat& samplingMap, SAMPLING samplingType, int d);


	std::vector<cv::Mat> samplingMapLUTs;
	cv::Mat samplingMap;
	std::vector<cv::Point> sampleLUT;//not used
};