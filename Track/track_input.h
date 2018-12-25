//
// Created by jerett on 2018-12-24.
//

/**
 *
 *  Implements track img input, support video input and img seq dir input.
 *
 */


#pragma once

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

namespace cd {

class TrackInput {

public:
    static std::shared_ptr<TrackInput> Create(const std::string &url);
    virtual ~TrackInput() = default;
    virtual bool Open() = 0;
    virtual bool Next(cv::Mat &out_frame) = 0;
    virtual double Get(int id) = 0;

};


class TrackVideoInput : public TrackInput {

public:
    explicit TrackVideoInput(const std::string &video_path) : video_path_(video_path) {
    }
    virtual ~TrackVideoInput() = default;

    bool Next(cv::Mat &out_frame) override;
    bool Open() override;
    double Get(int id) override;

private:
    std::string video_path_;
    cv::VideoCapture video_;
};


class TrackImgSeqDirInput : public TrackInput {

public:
    explicit TrackImgSeqDirInput(const std::string &dir) : dir_(dir) {
    }
    virtual ~TrackImgSeqDirInput() = default;

    bool Next(cv::Mat &out_frame) override;
    bool Open() override;
    double Get(int id) override;

private:
    std::string dir_;
    std::vector<std::string> files_;
    double width_ = 0;
    double height_ = 0;
    int index_ = 0;
    const double fps_ = 30;
};

//class TrackImageSeqDirInput : public TrackInput {
//public:
//    TrackImageSeqDirInput(const std::string &seq_dir) : seq_dir_(seq_dir) {
//    }
//    virtual ~TrackImageSeqDirInput() = default;
//
//    bool Next(cv::Mat &out_frame) override;
//    bool Open() override;
//
//private:
//    std::string seq_dir_;
//    cv::VideoCapture video_;
//};

}

