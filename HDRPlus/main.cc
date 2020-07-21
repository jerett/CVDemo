
#include <iostream>
#include <opencv2/opencv.hpp>
#include <direct.h>
#include <fstream>
#include "tinydir.h"
#include "DNGFile.h"
#include "ISP.h"
#include "HDRPlus.h"


int TestISP(const std::string &path) {
	DNGFile file;
	file.Read(path);
	file.ReadMetadata();
	// test GetPixelType
	//std::cout << "(0,0):" << file.GetPixelType(0, 0) << std::endl;
	//std::cout << "(0,1):" << file.GetPixelType(0, 1) << std::endl;
	//std::cout << "(1,0):" << file.GetPixelType(1, 0) << std::endl;
	//std::cout << "(1,1):" << file.GetPixelType(1, 1) << std::endl;

	cv::Mat bgrImg;
	RAW2JPG(file.GetImage(), *file.GetMetadata(), bgrImg);
	return 0;
}


void CopyFile2(const std::string &srcFile, const std::string &dstFile) {
	std::ifstream srcStream(srcFile, std::ios::binary);
	std::ofstream destStream(dstFile, std::ios::binary);
	destStream << srcStream.rdbuf();
}

int TestStack(const std::string &path) {
	std::vector<std::shared_ptr<DNGFile>> dngFiles;
	tinydir_dir dir;

	//const std::string dirPath = 
	//	"E:\\dataset\\HDR+BurstPhotographyDataset\\20171106_subset\\bursts\\";
	//const std::string prefix = "4KK2_20150823_152106_985\\";
	//const std::string prefix = "c1b1_20150424_203526_981\\";
	//const std::string prefix = "5a9e_20141010_173648_475\\";
	//const std::string prefix = "5a9e_20141007_173602_240\\";
	//const std::string prefix = "6G7M_20150321_151858_636\\";
	//const std::string prefix = "0039_20141006_110442_472\\";
	//const std::string path = dirPath + prefix;

	//const std::string path = ".";
	
	tinydir_open(&dir, path.c_str());
	while (dir.has_next) {
		tinydir_file file;
		tinydir_readfile(&dir, &file);

		if (strcmp(file.extension, "dng") == 0) {
			//std::cout << file.name;
			std::shared_ptr<DNGFile> dngf = std::make_shared<DNGFile>();
			int ret = dngf->Read(file.path);
			if (ret != 0) {
				//std::cerr << "Read dng file:" << file.name << std::endl;
				return -1;
			} else {
				//std::cout << "Read dng file:" << file.path << std::endl;
				dngf->ReadMetadata();
				auto metadata = dngf->GetMetadata();

				//BayerPattern bayerPattern = (BayerPattern)(ret);
				//std::cout << " BayerPattern:" << metadata->GetBayerPattern() << std::endl
				//	<< " ISO:" << metadata->GetISO() << std::endl
				//	<< " Shutter:" << metadata->GetShutter() << std::endl
				//	<< " BlackLevel Rows:" << metadata->GetBlackLevelRows() << "  Cols:" << metadata->GetBlackLevelCols() << std::endl
				//	<< " BlackLevels:\n" << metadata->GetBlackLevels() << std::endl
				//	<< " WhiteLevel:" << metadata->GetWhiteLevel() << std::endl
				//	<< " WBGain:" << metadata->GetWBGain() << std::endl
				//	<< " ColorMatrix:\n" << metadata->GetCameraFromXYZColorMatrix() << std::endl
				//	;
			}
			dngFiles.push_back(dngf);
		}
		tinydir_next(&dir);
	}
	std::vector<cv::Mat> rawImgs;
	std::vector<std::shared_ptr<DNGMetadata>> imgsMeatadata;
	for (const auto& f : dngFiles) {
		rawImgs.push_back(f->GetImage());
		imgsMeatadata.push_back(f->GetMetadata());
	}
	cv::Mat bgrImg;
	HDRPlus(rawImgs, imgsMeatadata, bgrImg);
	return 0;
}
 


void RunGoogleBurstDataset(const std::string& datasetPath) {
	const std::string burstPath = datasetPath + "bursts/";
	const std::string googleResultPath = datasetPath + "results_20171023/";
	_mkdir("results");
	tinydir_dir dir;
	tinydir_open(&dir, burstPath.c_str());

	std::string cwd = getcwd(0, 0);
	std::cout << "cwd:" << cwd << std::endl;
	int cnt = 0;
	while (dir.has_next) {
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0) {
			//std::cout << "discard:" << file.path << std::endl;
		} else {
			const std::string resultPath = "results/" + std::string(file.name);
			const std::string googleResultFile = googleResultPath + std::string(file.name) + "/final.jpg";

			std::string cwdP = cwd + "\\" + resultPath;
			_mkdir(cwdP.c_str());
			chdir(cwdP.c_str());
			std::cout << "chdir " << cwdP << std::endl;
			std::cout << "Stack " << file.path << std::endl;
			TestStack(file.path);
			CopyFile2(googleResultFile, "final.jpg");
			++cnt;
			if (cnt >= 20) break;
		}
		tinydir_next(&dir);
	}
}


int main(int argc, char *argv[]) {
	if (argc < 2) {
		std::cerr << "usage: ./hdrPlus <googleHdrDatasetPath>" << std::endl;
		return -1;
	}
	std::cout << "hello, hdrplus" << std::endl;
	const std::string datasetPath(argv[1]);

	//std::string dngPath = datasetPath + "/20171106_subset/bursts/4KK2_20150823_152106_985/payload_N000.dng";
	//TestISP(dngPath);

	//std::string shotPath = datasetPath + "/20171106_subset/bursts/4KK2_20150823_152106_985/";
	//TestStack(shotPath);

	const std::string burstsPath = datasetPath + "/20171106_subset/";
	RunGoogleBurstDataset(burstsPath);
	return 0;
}


