
#include "DNGFile.h"

#include <dng_file_stream.h>
#include <dng_auto_ptr.h>
#include <dng_host.h>
#include <dng_info.h>
#include <dng_errors.h>
#include <dng_camera_profile.h>


int DNGFile::Read(const std::string& path) {
	dng_file_stream stream(path.c_str());

	dng_host host;
	host.SetPreferredSize(0);
	host.SetMinimumSize(0);
	host.SetMaximumSize(0);
	host.ValidateSizes();

	if (host.MinimumSize())
		host.SetForPreview(true);
	dng_info info;
	info.Parse(host, stream);
	info.PostParse(host);
	if (!info.IsValidDNG()) {
		return dng_error_bad_format;
	}
	dng_.Reset(host.Make_dng_negative());
	dng_->Parse(host, stream, info);
	dng_->PostParse(host, stream, info);
	dng_->ReadStage1Image(host, stream, info);
	if (info.fMaskIndex != -1) {
		dng_->ReadTransparencyMask(host, stream, info);
	}
	dng_->ValidateRawImageDigest(host);

	const dng_image& dngRaw = dng_->RawImage();
	int pixelType = dngRaw.PixelType();
	int cols = dngRaw.Width();
	int rows = dngRaw.Height();
	int planes = dngRaw.Planes();
	int pixelSize = dngRaw.PixelSize();
	//std::cout << "read:" << path << "," << std::endl
	//	<< " pixelType:" << pixelType
	//	<< " cols:" << cols
	//	<< " rows:" << rows
	//	<< " planes:" << planes
	//	<< " pixelSize:" << pixelSize
	//	<< std::endl;

	int cvType;
	if (pixelType == ttShort) {
		cvType = CV_16UC(planes);
	} else if (pixelType == ttSShort) {
		cvType = CV_16SC(planes);
	} else if (pixelType == ttByte) {
		cvType = CV_8UC(planes);
	} else if (pixelType == ttLong) {
		cvType = CV_32SC(planes);
	} else {
		std::cerr << "unsupported pixelType:" << pixelType << std::endl;
	}
	img_.create(cv::Size(cols, rows), cvType);

	dng_pixel_buffer buffer(dng_rect(0, 0, rows, cols), 0, planes, pixelType, pcInterleaved, img_.data);
	dngRaw.Get(buffer, dng_image::edge_none);
	return dng_error_none;
}

int DNGFile::ReadMetadata() {
	metadata_ = std::make_shared<DNGMetadata>();
	int ret = ReadBayerPattern();
	if (ret == -1) return ret;
	ret = ReadISO();
	if (ret == -1) return ret;
	ret = ReadShutter();
	if (ret == -1) return ret;
	ret = ReadBlackLevels();
	if (ret == -1) return ret;
	ret = ReadWhiteLevel();
	if (ret == -1) return ret;
	ret = ReadWBGain();
	if (ret == -1) return ret;
	ret = ReadCameraFromXYZColorMatrix();
	if (ret == -1) return ret;
	return 0;
}

int DNGFile::ReadBayerPattern() {
	const dng_mosaic_info *mosaic_info = dng_->GetMosaicInfo();
	const auto bayerPattern_ = mosaic_info->fCFAPattern;
	
	// r=0, g=1, b=2
	if (bayerPattern_[0][0] == 1 && bayerPattern_[0][1] == 0) {
		metadata_->bayerPattern_ = GR_BG;
		metadata_->filter_ = 0x61616161;
	} else if (bayerPattern_[0][0] == 0 && bayerPattern_[0][1] == 1) {
		metadata_->bayerPattern_ = RG_GB;
		metadata_->filter_ = 0x94949494;
	} else if (bayerPattern_[0][0] == 2 && bayerPattern_[0][1] == 1) {
		metadata_->bayerPattern_ = BG_GR;
		metadata_->filter_ = 0x16161616;
	} else if (bayerPattern_[0][0] == 1 && bayerPattern_[0][1] == 2) {
		metadata_->bayerPattern_ = GB_RG;
		metadata_->filter_ = 0x49494949;
	} else {
		metadata_->filter_ = 0;
		return -1;
	}
}

int DNGFile::ReadISO() {
	metadata_->iso_ = dng_->GetExif()->fISOSpeedRatings[0];
	return 0;
}

int DNGFile::ReadShutter() {
	metadata_->shutter_ = dng_->GetExif()->fExposureTime.As_real64();
	return 0;
}

int DNGFile::ReadWhiteLevel() {
	auto linearizationInfo = dng_->GetLinearizationInfo();
	metadata_->whiteLevel_ = linearizationInfo->fWhiteLevel[0];
	return 0;
}

int DNGFile::ReadWBGain() {
	const auto &wb = dng_->CameraNeutral();
	metadata_->WBGain_(0) = static_cast<float>(wb[0]);
	metadata_->WBGain_(1) = static_cast<float>(wb[1]);
	metadata_->WBGain_(2) = static_cast<float>(wb[2]);
	return 0;
}

int DNGFile::ReadCameraFromXYZColorMatrix() {
	const dng_camera_profile &cameraProfile = dng_->ProfileByIndex(0);
	auto m = const_cast<dng_matrix&>(cameraProfile.ColorMatrix1());

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			metadata_->cameraFromXYZColorMatrix_(i, j) = m[i][j];
		}
	}
	return 0;
}

int DNGFile::ReadBlackLevels() {
	auto linearizationInfo = dng_->GetLinearizationInfo();
	const double black00 = linearizationInfo->fBlackLevel[0][0][0];
	const double black01 = linearizationInfo->fBlackLevel[0][1][0];
	const double black10 = linearizationInfo->fBlackLevel[1][0][0];
	const double black11 = linearizationInfo->fBlackLevel[1][1][0];

	metadata_->blackLevelRepeatCols_ = linearizationInfo->fBlackLevelRepeatCols;
	metadata_->blackLevelRepeatRows_ = linearizationInfo->fBlackLevelRepeatRows;

	if (linearizationInfo->fBlackLevelRepeatCols == 1 && linearizationInfo->fBlackLevelRepeatRows == 1) {
		//blackLevels_.set(black00);
		metadata_->blackLevels_ = cv::Matx22f::all(black00);
	} else {
		metadata_->blackLevels_(0, 0) = black00;
		metadata_->blackLevels_(0, 1) = black01;
		metadata_->blackLevels_(1, 0) = black10;
		metadata_->blackLevels_(1, 1) = black11;
	}
	return 0;
}



