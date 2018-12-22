//
// Created by jerett on 2018-12-21.
//

/*
 * demostrate and compare some track algorithm.
 *
 */

#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <boost/filesystem.hpp>

namespace bf = boost::filesystem;
using namespace cv;


Rect2d ChooseRect(Mat &picture) {
    return selectROI("track", picture, false);
}

Ptr<Tracker> CreateTracker(const std::string &tracker_type) {
    Ptr<Tracker> tracker;
    if (tracker_type == "Boosting") {
        tracker = TrackerBoosting::create();
    } else if (tracker_type == "KCF") {
        TrackerKCF::Params p;
        p.desc_npca = TrackerKCF::GRAY | TrackerKCF::CN;
//        p.desc_npca = TrackerKCF::CN;
        tracker = TrackerKCF::create(p);
    } else if (tracker_type == "MIL") {
        tracker = TrackerMIL::create();
    } else if (tracker_type == "TLD") {
        tracker = TrackerTLD::create();
    } else if (tracker_type == "MedianFlow") {
        tracker = TrackerMedianFlow::create();
    } else if (tracker_type == "GOTURN") {
        tracker = TrackerGOTURN::create();
    } else if (tracker_type == "MOSSE") {
        tracker = TrackerMOSSE::create();
    }
    assert(tracker != nullptr);
    return tracker;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        CV_LOG_ERROR(NULL, "usage: track <track_type> <src_file> <out_dir>");
        return -1;
    }
    std::string tracker_type(argv[1]);
    std::string in_file(argv[2]);
    std::string out_dir(argv[3]);


    VideoCapture video;
    if (!video.open(in_file)) {
        CV_LOG_ERROR(NULL, "open " << in_file << " failed.")
        return -1;
    }
    double src_fps = video.get(cv::CAP_PROP_FPS);
    double width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    Size src_size(width, height);
    CV_LOG_INFO(NULL, "src video fps:" << src_fps << " width:" << width << " height:" << height);
    bf::path in_path(in_file);
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
    if (!video.read(first_frame)) {
        CV_LOG_ERROR(NULL, "read first frame failed.");
        return -1;
    }
    auto rect = ChooseRect(first_frame);
    CV_LOG_INFO(NULL, "choose rect " << rect);

    bool r = tracker->init(first_frame, rect);
    if (!r) {
        CV_LOG_ERROR(NULL, "init tracker failed.");
        return -1;
    }

    cv::Mat frame;
    cv::Rect2d track_box;
    int frame_cnt = 0;
    while (video.read(frame)) {
        ++frame_cnt;
        r = tracker->update(frame, track_box);
        if (!r) {
            CV_LOG_ERROR(NULL, "update tracker failed.");
            // reinit
            tracker = CreateTracker(tracker_type);
            track_box = ChooseRect(frame);
            r = tracker->init(frame, track_box);
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
        if(k == 27 || k == 81) {
            break;
        }
    }
    video_writer.release();
    return 0;
}
