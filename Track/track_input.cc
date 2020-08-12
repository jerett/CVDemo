//
// Created by jerett on 2018-12-24.
//

#include "track_input.h"
#include "tinydir.h"
#include <string>
//#include <boost/filesystem.hpp>

//namespace fs = boost::filesystem;

namespace cd {

std::shared_ptr<TrackInput> TrackInput::Create(const std::string &url) {
    tinydir_file f;
    if (tinydir_file_open(&f, url.c_str()) != 0) return nullptr;
    //tinydir_file file;
    //tinydir_readfile(&dir, &file);
    //if (dir.has_next)
    //fs::path p(url);
    std::shared_ptr<TrackInput> track_input;
    if (f.is_dir) {
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
    tinydir_dir dir;
    if (tinydir_open(&dir, dir_.data()) != 0) {
        return false;
    }
    //fs::path p(dir_);
    //fs::directory_iterator end_itr;

    std::vector<std::string> files;
    // cycle through the directory
    while (dir.has_next) {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        tinydir_file file;
        tinydir_readfile(&dir, &file);
        //std::cout << "ext:" << file.extension << std::endl;
        if (file.is_reg && strcmp(file.extension, "jpg") == 0) {
            // assign current file name to current_file and echo it out to the console.
            std::string current_file = file.path;
            files.push_back(current_file);
//            std::cout << current_file << std::endl;
        }
        tinydir_next(&dir);
    }
    tinydir_close(&dir);
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