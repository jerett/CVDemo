//
// Created by jerett on 2018-12-24.
//

#include "track_input.h"
#include <string>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace cd {

std::shared_ptr<TrackInput> TrackInput::Create(const std::string &url) {
    fs::path p(url);
    std::shared_ptr<TrackInput> track_input;
    if (fs::is_directory(p)) {
        track_input = std::make_shared<TrackImgSeqDirInput>(url);
    } else {
        track_input = std::make_shared<TrackVideoInput>(url);
    }
    return track_input;
}


bool TrackVideoInput::Open() {
    return video_.open(video_path_);
}

bool TrackVideoInput::Next(cv::Mat &out_frame) {
    return video_.read(out_frame);
}

double TrackVideoInput::Get(int id) {
    return video_.get(id);
}


bool TrackImgSeqDirInput::Open() {
    fs::path p(dir_);
    fs::directory_iterator end_itr;

    std::vector<std::string> files;
    // cycle through the directory
    for (fs::directory_iterator itr(p); itr != end_itr; ++itr) {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (is_regular_file(itr->path()) && itr->path().extension() == ".jpg") {
            // assign current file name to current_file and echo it out to the console.
            std::string current_file = itr->path().string();
            files.push_back(current_file);
//            std::cout << current_file << std::endl;
        }
    }
    std::sort(files.begin(), files.end());
    files_ = std::move(files);
    if (files_.empty()) {
        return false;
    } else {
        cv::Mat first_img = cv::imread(files_[0]);
        width_ = first_img.cols;
        height_ = first_img.rows;
        return true;
    }
}

double TrackImgSeqDirInput::Get(int id) {
    if (id == cv::CAP_PROP_FRAME_WIDTH) {
        return width_;
    } else if (id == cv::CAP_PROP_FRAME_HEIGHT) {
        return height_;
    } else if (id == cv::CAP_PROP_FPS) {
        return fps_;
    } else {
        return -1;
    }
}

bool TrackImgSeqDirInput::Next(cv::Mat &out_frame) {
    if (index_ < files_.size()) {
        out_frame = cv::imread(files_[index_]);
        ++index_;
        return true;
    } else {
        return false;
    }
}


}