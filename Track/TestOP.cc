 
#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat TemplateMatch(const cv::Mat& img, const cv::Mat& tmpl) {
	cv::Mat result;
	//int method = cv::TM_SQDIFF;
	//int method = cv::TM_SQDIFF_NORMED;
	//int method = cv::TM_CCORR;
	int method = cv::TM_CCORR_NORMED;
	//int method = cv::TM_CCOEFF;
	//int method = cv::TM_CCOEFF_NORMED;
	//cv::matchTemplate(shelfImg, templateImg, result, cv::TM_CCORR);
	//cv::matchTemplate(shelfImg, templateImg, result, cv::TM_CCORR_NORMED);
	cv::matchTemplate(img, tmpl, result, method);
	return result;
}


cv::Point FFTMatch(const cv::Mat& img, const cv::Mat& tmpl) {
	cv::Mat F, H;

	cv::Mat img32f;
	img.convertTo(img32f, CV_32FC1);
	cv::dft(img32f, H, cv::DFT_COMPLEX_OUTPUT);

	//cv::Mat tmp;
	//cv::dft(H, tmp, cv::DFT_INVERSE | cv::DFT_SCALE);

	cv::Mat tmpl32f;
	tmpl.convertTo(tmpl32f, CV_32FC1);
	cv::dft(tmpl32f, F, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat G;
	cv::mulSpectrums(H, F, G, 0, true);

	cv::Mat result;
	cv::idft(G, result, cv::DFT_REAL_OUTPUT);

	cv::Mat mag = result.clone();
	int cx = mag.cols / 2;
    int cy = mag.rows/2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center
    cv::Mat tmp;
    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

	cv::Point maxLoc;
	double maxVal;
	minMaxLoc(mag, NULL, &maxVal, NULL, &maxLoc);

	cv::Point center(cx, cy);
	cv::Point shift = center - maxLoc;

	return shift;
}

using namespace cv;
void TestDFT(const cv::Mat &I) {
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix
	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);
	waitKey();
}


int main(int argc, char* argv[]) {
	// load as gray
	//cv::Mat templateImg = cv::imread("template2.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat templateImg = cv::imread("shelf2.jpg", cv::IMREAD_GRAYSCALE);
	//cv::Mat templateImg = cv::imread("shelf3.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat tmpl32f;
	templateImg.convertTo(tmpl32f, CV_32FC1);

	cv::Mat img = cv::imread("shelf.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img32f;
	img.convertTo(img32f, CV_32FC1);

	cv::Mat textNImg = cv::imread("imageTextN.png", cv::IMREAD_GRAYSCALE);
	cv::Mat textRImg = cv::imread("imageTextR.png", cv::IMREAD_GRAYSCALE);
	TestDFT(textNImg);
	TestDFT(textRImg);

	//cv::Mat result = TemplateMatch(img, templateImg);
	//cv::Mat result2 = FFTMatch(img, templateImg);

	//double maxVal;
	//cv::Point maxLoc;
	//cv::minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

	//cv::Point shift = cv::phaseCorrelate(tmpl32f, img32f);

	cv::Point shift = FFTMatch(img, templateImg);
	std::cout << "fftMatch shift:" << shift << std::endl;
	cv::Point shift2 = cv::phaseCorrelate(img32f, tmpl32f);
	std::cout << "shift:" << shift2 << std::endl;
	
	return 0;
}