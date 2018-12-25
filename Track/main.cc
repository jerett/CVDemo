//
// Created by jerett on 2018-12-21.
//

/*
 * demostrate and compare some track algorithm.
 *
 */

#include <string>
#include <map>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <boost/filesystem.hpp>
#include "tracker.h"
#include "track_input.h"

namespace bf = boost::filesystem;
using namespace cv;

Rect2d ChooseRect(Mat &picture) {
    return selectROI("track", picture, false);
}

Ptr<cd::Tracker> CreateTracker(const std::string &tracker_type) {
    std::map<std::string, cd::Tracker::Algorithm> str_algorithm_map = {
        {"Boosting", cd::Tracker::Algorithm::Boosting},
        {"KCF", cd::Tracker::Algorithm::KCF},
        {"MIL", cd::Tracker::Algorithm::MIL},
        {"TLD", cd::Tracker::Algorithm::TLD},
        {"MedianFlow", cd::Tracker::Algorithm::MedianFlow},
        {"GOTURN", cd::Tracker::Algorithm::GOTURN},
        {"MOSSE", cd::Tracker::Algorithm::MOSSE},
        {"Staple", cd::Tracker::Algorithm::Staple},
    };
    Ptr<cd::Tracker> tracker;
    auto itr = str_algorithm_map.find(tracker_type);
    if (itr == str_algorithm_map.end()) {
        abort();
    } else {
        CV_LOG_DEBUG(NULL, "algorithm:" << itr->second);
        tracker = cd::Tracker::Create(itr->second);
    }
    assert(tracker != nullptr);
    return tracker;
}

int track(const std::string &src,
          const std::string &out_dir,
          const std::string &tracker_type) {
    auto track_input = cd::TrackInput::Create(src);
    if (!track_input->Open()) {
        CV_LOG_ERROR(NULL, "open " << src << " failed.")
        return -1;
    }
    double src_fps = track_input->Get(cv::CAP_PROP_FPS);
    double width = track_input->Get(cv::CAP_PROP_FRAME_WIDTH);
    double height = track_input->Get(cv::CAP_PROP_FRAME_HEIGHT);
    Size src_size(width, height);
    CV_LOG_INFO(NULL, "src video fps:" << src_fps << " width:" << width << " height:" << height);
    bf::path in_path(src);
    std::string basename = bf::basename(in_path);
    CV_LOG_INFO(NULL, basename);

    auto tracker = CreateTracker(tracker_type);
    cv::VideoWriter video_writer;
    // test h264 encode first.
    bf::path out_path = bf::path(out_dir) / (basename + "_" + tracker_type + ".mp4");
    CV_LOG_INFO(NULL, "out path:" << out_path);

    if (!video_writer.open(out_path.string(), cv::VideoWriter::fourcc('H', '2', '6', '4'), src_fps, src_size)) {
        CV_LOG_ERROR(NULL, "using H264 writer failed.");
        return -1;
    }

    cv::Mat first_frame;
    if (!track_input->Next(first_frame)) {
        CV_LOG_ERROR(NULL, "read first frame failed.");
        return -1;
    }
    auto rect = ChooseRect(first_frame);
    CV_LOG_INFO(NULL, "choose rect " << rect);

    bool r = tracker->Init(first_frame, rect);
    if (!r) {
        CV_LOG_ERROR(NULL, "init tracker failed.");
        return -1;
    }

    cv::Mat frame;
    cv::Rect2d track_box;
    int frame_cnt = 0;
    while (track_input->Next(frame)) {
        ++frame_cnt;
        r = tracker->Update(frame, track_box);
        if (!r) {
            CV_LOG_ERROR(NULL, "update tracker failed.");
            // reinit
            tracker = CreateTracker(tracker_type);
            track_box = ChooseRect(frame);
            r = tracker->Init(frame, track_box);
            if (!r) {
                CV_LOG_ERROR(NULL, "reinit tracker failed.");
                return -1;
            }
        }
        cv::rectangle(frame, track_box, cv::Scalar(255, 0, 0), 2);
        CV_LOG_INFO(NULL, "read " << frame.rows << " " << frame.cols << ", frame_cnt:" << frame_cnt);
        cv::imshow("track", frame);
        video_writer.write(frame);
        int k = cv::waitKey(1);
        // press 'esc' or 'q'
        if (k == 27 || k == 'q' || k == 'Q') {
            break;
        }
    }
    video_writer.release();
    return 0;
}

int main(int argc, char *argv[]) {
    const String keys =
        "{help h usage ? |      | print this message   }"
        "{type t         |Staple| track algorithm type }"
        "{input i        |<none>| img seq dir or video path for track }"
        "{out_dir o      |.     | output_dir for write esult mp4, name will be auto generated. }";
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string src = parser.get<String>("input");
    std::string out_dir(parser.get<String>("out_dir"));
    std::string tracker_type(parser.get<String>("type"));
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    return track(src, out_dir, tracker_type);
}
