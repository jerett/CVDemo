
#pragma once

#include <string>
#include <dng_auto_ptr.h>
#include <opencv2/opencv.hpp>

class dng_negative;

enum BayerPattern {
	GR_BG,
	RG_GB,
	BG_GR,
	GB_RG,
};

enum PixelType {
	R,
	G,
	B
};


class DNGMetadata {
	friend class DNGFile;
public:
	int GetISO() const {
		return iso_;
	}

	float GetShutter() const {
		return shutter_;
	}

	BayerPattern GetBayerPattern() const {
		return bayerPattern_;
	}

	cv::Matx22f GetBlackLevels() const {
		return blackLevels_;
	}

	int GetBlackLevelRows() const {
		return blackLevelRepeatRows_;
	}

	int GetBlackLevelCols() const {
		return blackLevelRepeatCols_;
	}

	int GetWhiteLevel() const {
		return whiteLevel_;
	}

	cv::Vec3f GetWBGain() const {
		return WBGain_;
	}

	cv::Matx33f GetCameraFromXYZColorMatrix() const {
		return cameraFromXYZColorMatrix_;
	}

	// copy from dcraw
	PixelType GetPixelType(int row, int col) const {
		const uint32_t type = filter_ >> ((((row) << 1 & 14) + ((col) & 1)) << 1) & 3;
		return PixelType(type);
	}

	cv::Matx33f GetXYZFromSRGBColorMatrix() const {
		static const cv::Matx33f XYZFromRGB =
		{ 0.412453f, 0.357580f, 0.180423f, 0.212671f, 0.715160f, 0.072169f, 0.019334f, 0.119193f, 0.950227f };
		return XYZFromRGB;
	}

private:
	int iso_;
	float shutter_;

	BayerPattern bayerPattern_;
	uint32_t filter_;

	int blackLevelRepeatCols_;
	int blackLevelRepeatRows_;
	cv::Matx22f blackLevels_;
	cv::Matx33f cameraFromXYZColorMatrix_;

	double whiteLevel_;
	// r, g, b gain
	cv::Vec3f WBGain_;

};


class DNGFile {

public:
	DNGFile() = default;
	~DNGFile() = default;

	int Read(const std::string &path);
	int ReadMetadata();


	cv::Mat GetImage() const {
		return img_;
	}

	std::shared_ptr<DNGMetadata> GetMetadata() const {
		return metadata_;
	}


private:
	int ReadBayerPattern();
	int ReadISO();
	int ReadShutter();
	int ReadBlackLevels();
	int ReadWhiteLevel();
	int ReadWBGain();
	int ReadCameraFromXYZColorMatrix();

private:
	AutoPtr<dng_negative> dng_;
	std::shared_ptr<DNGMetadata> metadata_;
	cv::Mat img_;

	//int iso_;
	//float shutter_;

	//BayerPattern bayerPattern_;
	//uint32_t filter_;

	//int blackLevelRepeatCols_;
	//int blackLevelRepeatRows_;
	//cv::Matx22f blackLevels_;
	//cv::Matx33f cameraFromXYZColorMatrix_;

	//double whiteLevel_;
	//// r, g, b gain
	//cv::Vec3f WBGain_;
};
