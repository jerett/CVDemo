
#include <opencv2/opencv.hpp>


class MOSSETracker {
public:
	MOSSETracker();

	//cv::Mat 
	bool Init(const cv::Mat &img, const cv::Rect2d &box);
	bool Update(const cv::Mat& img, cv::Rect2d& outbox);

private:
	float Detect(const cv::Mat& img, cv::Rect2d &outbox);
	bool Train(const cv::Mat& img, cv::Rect2d &outbox);

//private:
//	bool Update_(const cv::Mat& img, cv::Rect2d& outbox);

private:
	cv::Point2f center_;
	cv::Size size_;
	cv::RNG rng;
	cv::Mat G_;
	cv::Mat Hconj_;
	cv::Mat A_;
	cv::Mat B_;
	double rate_ = .4f;
	int frameCnt = 0;
	//bool Train();
	//bool Update();


};